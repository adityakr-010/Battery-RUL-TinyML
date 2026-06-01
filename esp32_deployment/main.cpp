// Dependencies
#include <math.h>
#include <Arduino.h>
#define MAX_SAMPLES 3600
#define SEQ_LEN 10
#define NUM_FEATURES 11
#include "model_finalVersion.h"

#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

#include <WiFi.h>
#include <WebServer.h>

// Wifi configuration
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// Global variables
WebServer server(80);

struct Telemetry {
    float rul;
    float capacity;
    float max_temp;
    float energy;
};

Telemetry latest_data = {-1.0f, 0.0f, 0.0f, 0.0f};

SemaphoreHandle_t dataReadySemaphore;
QueueHandle_t resultQueue;
TaskHandle_t InferenceTaskHandle;

// TFLite variables
tflite::MicroErrorReporter micro_error_reporter;
constexpr int kTensorArenaSize = 10 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* model_ptr;
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;
static tflite::MicroMutableOpResolver<6> resolver;

// Data buffers
float sequence[SEQ_LEN][NUM_FEATURES];
int cycleCounter = 0;
int validCycles = 0;

float tBuf[MAX_SAMPLES];
float vBuf[MAX_SAMPLES];
float qBuf[MAX_SAMPLES];
float TBuf[MAX_SAMPLES];

const float scalerMean[10] = {
    636.805105, 41.5233069, 0.0358721535, 40.8174610, 41.7140761, 
    3.73887712, 0.234922974, 3759.21033, 2497649.06, -2.67717645
};

const float scalerScale[10] = {
    51.0259897, 6.59848403, 0.00287767231, 0.457367752, 0.580503248, 
    0.00712888923, 0.00493979823, 356.321222, 228296.459, 0.350837784
};

int sampleCount = 0;
const float YMAX_MASTER = 65.0f; 

// Feature computation
Telemetry computeFeatures() {
    Telemetry result = {-1.0f, 0.0f, 0.0f, 0.0f};

    float duration = tBuf[sampleCount - 1] - tBuf[0];
    float capacity = fabs(qBuf[sampleCount - 1] - qBuf[0]);
    
    float avgTemp = 0;
    for(int i=0; i<sampleCount; i++) { avgTemp += TBuf[i]; }
    avgTemp /= sampleCount;

    float maxTemp = TBuf[0];
    for(int i=1; i<sampleCount; i++) {
        if(TBuf[i] > maxTemp) maxTemp = TBuf[i];
    }

    float voltageMean = 0;
    for(int i=0; i<sampleCount; i++) { voltageMean += vBuf[i]; }
    voltageMean /= sampleCount;
    
    float voltageStd = 0;
    for(int i=0; i<sampleCount; i++) {
        float diff = vBuf[i] - voltageMean;
        voltageStd += diff * diff;
    }
    voltageStd /= sampleCount;
    voltageStd = sqrt(voltageStd);
    
    float energy = 0;
    for(int i=0; i<sampleCount-1; i++) {
        float y1 = vBuf[i] * fabs(qBuf[i]);
        float y2 = vBuf[i+1] * fabs(qBuf[i+1]);
        float dt = tBuf[i+1] - tBuf[i];
        energy += 0.5f * (y1 + y2) * dt;
    }
    
    float dqdvPeak = 0.0f;
    float dqdvArea = 0.0f;

    float* raw_dqdv = (float*)malloc(MAX_SAMPLES * sizeof(float));
    float* smoothed_dqdv = (float*)malloc(MAX_SAMPLES * sizeof(float));

    if (raw_dqdv != NULL && smoothed_dqdv != NULL) {
        int dqdvCount = 0;
        for(int i=0; i<sampleCount-1; i++) {
            float dv = vBuf[i+1] - vBuf[i];
            if(fabs(dv) > 1e-5f) {
                raw_dqdv[dqdvCount++] = (qBuf[i+1] - qBuf[i]) / dv;
            }
        }

        if(dqdvCount > 10) {
            const float sg_coeffs[11] = {-36.0f, 9.0f, 44.0f, 69.0f, 84.0f, 89.0f, 84.0f, 69.0f, 44.0f, 9.0f, -36.0f};
            const float sg_norm = 429.0f;

            for(int i=0; i<dqdvCount; i++) {
                float sum = 0.0f;
                for(int j=-5; j<=5; j++) {
                    int idx = i + j;
                    if(idx < 0) idx = 0; 
                    if(idx >= dqdvCount) idx = dqdvCount - 1;
                    sum += raw_dqdv[idx] * sg_coeffs[j + 5];
                }
                smoothed_dqdv[i] = sum / sg_norm;
                if(fabs(smoothed_dqdv[i]) > dqdvPeak) dqdvPeak = fabs(smoothed_dqdv[i]);
            }
            for(int i=0; i<dqdvCount-1; i++) {
                dqdvArea += 0.5f * (fabs(smoothed_dqdv[i]) + fabs(smoothed_dqdv[i+1]));
            }
        }
    }
    if (raw_dqdv) free(raw_dqdv);
    if (smoothed_dqdv) free(smoothed_dqdv);

    float vMin = vBuf[0];
    float vMax = vBuf[0];
    for(int i=1; i<sampleCount; i++) {
        if(vBuf[i] < vMin) vMin = vBuf[i];
        if(vBuf[i] > vMax) vMax = vBuf[i];
    }
    
    const int NUM_BINS = 30;
    int hist[NUM_BINS] = {0};
    float binWidth = (vMax - vMin) / NUM_BINS;
    if(binWidth < 1e-12f) binWidth = 1e-12f;
    
    for(int i=0; i<sampleCount; i++) {
        int bin = (int)((vBuf[i] - vMin) / binWidth);
        if(bin >= NUM_BINS) bin = NUM_BINS - 1;
        if(bin < 0) bin = 0;
        hist[bin]++;
    }
    
    float entropy = 0.0f;
    for(int i=0; i<NUM_BINS; i++) {
        float p = (float)hist[i] / (sampleCount * binWidth);
        p += 1e-8f;
        entropy -= p * logf(p);
    } 

    float rawFeatures[10] = {capacity, energy, duration, avgTemp, maxTemp, voltageMean, voltageStd, dqdvPeak, dqdvArea, entropy};
    float features[NUM_FEATURES];
    float cycleIndex = (float)cycleCounter / 73.0f;

    for(int i=0; i<10; i++) {
        features[i] = (rawFeatures[i] - scalerMean[i]) / scalerScale[i];
    }
    features[10] = cycleIndex;

    for(int row=0; row<SEQ_LEN-1; row++) {
        for(int col=0; col<NUM_FEATURES; col++) {
            sequence[row][col] = sequence[row+1][col];
        }
    }
    for(int col=0; col<NUM_FEATURES; col++) {
        sequence[SEQ_LEN-1][col] = features[col];
    }
    
    cycleCounter++;
    if(validCycles < SEQ_LEN) validCycles++;
    
    if(validCycles == SEQ_LEN) {
        int idx = 0;
        for(int c=0; c<NUM_FEATURES; c++) {
            for(int r=0; r<SEQ_LEN; r++) {
                input->data.f[idx++] = sequence[r][c];
            }
        }

        TfLiteStatus invoke_status = interpreter->Invoke();

        if(invoke_status == kTfLiteOk) {
            result.rul = (output->data.f[0] * YMAX_MASTER);
            result.capacity = capacity;
            result.max_temp = maxTemp;
            result.energy = energy;
        } else {
            Serial.println("Inference Failed");
        }
    }
    return result; 
}

// Inference task
void inferenceTask(void * pvParameters) {
    for(;;) {
        if(xSemaphoreTake(dataReadySemaphore, portMAX_DELAY) == pdTRUE) {
            Telemetry computed_data = computeFeatures();
            xQueueSend(resultQueue, &computed_data, portMAX_DELAY);
        }
    }
}

// Setup function
void setup()
{
    Serial.begin(115200);
    delay(10000);

    WiFi.begin(ssid, password);
    Serial.print("Connecting to Wi-Fi");
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\n--- WIFI CONNECTED ---");
    Serial.print("Go to this IP in your browser: ");
    Serial.println(WiFi.localIP());

    server.on("/", []() {
        server.sendHeader("Cache-Control", "no-store, no-cache, must-revalidate");
        
        String html = "<!DOCTYPE html><html><head><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">";
        html += "<style>body{font-family:Arial,sans-serif;text-align:center;background-color:#1e1e2f;color:#fff;margin:50px 20px;} ";
        html += "h1{color:#fff;font-size:32px;margin-bottom:30px;} ";
        html += ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:20px;max-width:800px;margin:0 auto;} ";
        html += ".card{background:#2a2a40;padding:30px;border-radius:12px;box-shadow:0 4px 15px rgba(0,0,0,0.3);} ";
        html += ".rul-card{grid-column:1/-1;background:#3b3b58;} "; 
        html += ".val{font-size:48px;font-weight:bold;color:#4caf50;margin:10px 0;} ";
        html += ".val-sub{font-size:32px;color:#00bcd4;} ";
        html += ".label{font-size:16px;color:#aaa;text-transform:uppercase;letter-spacing:1px;}</style></head>";
        html += "<body><h1> RUL DASHBOARD</h1><div class=\"grid\">";
        
        html += "<div class=\"card rul-card\"><div class=\"label\">Remaining Useful Life</div>";
        html += "<div id=\"val-rul\" class=\"val\" style=\"color:#888;\">Buffering 10 Cycles...</div></div>";
        
        html += "<div class=\"card\"><div class=\"label\">Capacity Discharged</div><div id=\"val-cap\" class=\"val val-sub\">-- Ah</div></div>";
        html += "<div class=\"card\"><div class=\"label\">Max Core Temp</div><div id=\"val-tmp\" class=\"val val-sub\">-- &deg;C</div></div>";
        html += "<div class=\"card\"><div class=\"label\">Total Energy</div><div id=\"val-nrg\" class=\"val val-sub\">-- Wh</div></div>";
        html += "</div>";
        
        html += "<script>setInterval(function() {";
        html += "fetch('/data?t=' + new Date().getTime()).then(res => res.json()).then(data => {";
        html += "if(data.rul >= 0) {";
        html += "document.getElementById('val-rul').innerHTML = data.rul.toFixed(2) + ' <span style=\"font-size:24px;color:#aaa;\">Cycles</span>';";
        html += "document.getElementById('val-rul').style.color = '#4caf50';";
        html += "document.getElementById('val-cap').innerText = data.cap.toFixed(3) + ' mAh';";
        html += "document.getElementById('val-tmp').innerHTML = data.temp.toFixed(1) + ' &deg;C';";
        html += "document.getElementById('val-nrg').innerText = (data.nrg / 3600).toFixed(2) + ' Wh';"; 
        html += "}";
        html += "}).catch(err => console.log('ESP32 busy'));";
        html += "}, 10000);</script></body></html>"; 
        
        server.send(200, "text/html", html);
    });

    server.on("/data", []() {
        server.sendHeader("Cache-Control", "no-store, no-cache, must-revalidate");
        
        String json = "{\"rul\":" + String(latest_data.rul) + 
                      ",\"cap\":" + String(latest_data.capacity) + 
                      ",\"temp\":" + String(latest_data.max_temp) + 
                      ",\"nrg\":" + String(latest_data.energy) + "}";
        server.send(200, "application/json", json);
    });

    server.onNotFound([]() { server.send(204); }); 
    server.begin();

    dataReadySemaphore = xSemaphoreCreateBinary();
    resultQueue = xQueueCreate(1, sizeof(Telemetry));

    model_ptr = tflite::GetModel(model);
    resolver.AddConv2D();
    resolver.AddRelu();
    resolver.AddMean();
    resolver.AddFullyConnected();
    resolver.AddReshape();
    resolver.AddTranspose();

    interpreter = new tflite::MicroInterpreter(
        model_ptr, resolver, tensor_arena, kTensorArenaSize, &micro_error_reporter
    );

    if(interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("ALLOCATE FAILED");
        while(1);
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    xTaskCreatePinnedToCore(
        inferenceTask, "InferenceTask", 16384, NULL, 1, &InferenceTaskHandle, 0
    );
}

// Main loop
void loop()
{
    server.handleClient();

    if(Serial.available()) {
        auto line = Serial.readStringUntil('\n');
        line.trim();

        if(line == "END") {
            xSemaphoreGive(dataReadySemaphore);
            
            Telemetry final_data;
            if(xQueueReceive(resultQueue, &final_data, portMAX_DELAY) == pdTRUE) {
                if(final_data.rul >= 0.0f) {
                    latest_data = final_data;
                    Serial.print(">>> FINAL RUL: ");
                    Serial.print(final_data.rul, 2);
                    Serial.print(" | Cap: ");
                    Serial.print(final_data.capacity, 3);
                    Serial.print(" | Temp: ");
                    Serial.println(final_data.max_temp, 1);
                }
            }
            sampleCount = 0; 
            return;
        }

        float t, v, q, temp;
        int p1 = line.indexOf(',');
        int p2 = line.indexOf(',', p1+1);
        int p3 = line.indexOf(',', p2+1);

        if(p1 > 0 && p2 > 0 && p3 > 0) {
            t = line.substring(0, p1).toFloat();
            v = line.substring(p1+1, p2).toFloat();
            q = line.substring(p2+1, p3).toFloat();
            temp = line.substring(p3+1).toFloat();

            if(sampleCount < MAX_SAMPLES) {
                tBuf[sampleCount] = t;
                vBuf[sampleCount] = v;
                qBuf[sampleCount] = q;
                TBuf[sampleCount] = temp;
                sampleCount++;
            }
        }
    }
}

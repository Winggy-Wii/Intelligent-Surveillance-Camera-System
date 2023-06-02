#include "esp_camera.h"
#include <WiFi.h>
#include <ESP32Servo.h>
#include <ESPmDNS.h>
#include "PosVal.h"
#include <WiFiManager.h>
WiFiManager wm;
//
// WARNING!!! PSRAM IC required for UXGA resolution and high JPEG quality
//            Ensure ESP32 Wrover Module or other board with PSRAM is selected
//            Partial images will be transmitted if image exceeds buffer size
//

// Select camera model
//#define CAMERA_MODEL_WROVER_KIT // Has PSRAM
//#define CAMERA_MODEL_ESP_EYE // Has PSRAM
//#define CAMERA_MODEL_M5STACK_PSRAM // Has PSRAM
//#define CAMERA_MODEL_M5STACK_V2_PSRAM // M5Camera version B Has PSRAM
//#define CAMERA_MODEL_M5STACK_WIDE // Has PSRAM
//#define CAMERA_MODEL_M5STACK_ESP32CAM // No PSRAM
#define CAMERA_MODEL_AI_THINKER // Has PSRAM
//#define CAMERA_MODEL_TTGO_T_JOURNAL // No PSRAM

#include "camera_pins.h"

#define DUMMY_SERVO1_PIN 14     //We need to create 2 dummy servos.
#define DUMMY_SERVO2_PIN 15     //So that ESP32Servo library does not interfere with pwm channel and timer used by esp32 camera.



Servo dummyServo1;
Servo dummyServo2;
Servo panServo;
Servo tiltServo;

#define PAN_PIN 14
#define TILT_PIN 15

uint8_t HorPosVal = 0;
uint8_t VertPosVal = 0;

int freq = 4500;
const int ledChannel = 1;
const int something = 8;
const int ledPin = 12;  // 16 corresponds to GPIO16
bool lock = 1;

void startCameraServer();
void setup() {
  ledcSetup(ledChannel, freq, something);

  // attach the channel to the GPIO to be controlled
  ledcAttachPin(ledPin, ledChannel);
  for (int dutyCycle = 0; dutyCycle <= 255; dutyCycle++) {
    // changing the LED brightness with PWM
    ledcWrite(ledChannel, dutyCycle);
    delay(6);
  }

  for (int dutyCycle = 255; dutyCycle >= 0; dutyCycle--) {
    // changing the LED brightness with PWM
    ledcWrite(ledChannel, dutyCycle);
    delay(6);
  }
  dummyServo1.attach(DUMMY_SERVO1_PIN);
  dummyServo2.attach(DUMMY_SERVO2_PIN);
  panServo.attach(PAN_PIN);
  tiltServo.attach(TILT_PIN);
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println();

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // if PSRAM IC present, init with UXGA resolution and higher JPEG quality
  //                      for larger pre-allocated frame buffer.
  if (psramFound()) {
    config.frame_size = FRAMESIZE_UXGA;
    config.jpeg_quality = 10;
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_SVGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }

#if defined(CAMERA_MODEL_ESP_EYE)
  pinMode(13, INPUT_PULLUP);
  pinMode(14, INPUT_PULLUP);
#endif

  // camera init
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }

  sensor_t * s = esp_camera_sensor_get();
  // initial sensors are flipped vertically and colors are a bit saturated
  if (s->id.PID == OV3660_PID) {
    s->set_vflip(s, 1); // flip it back
    s->set_brightness(s, 1); // up the brightness just a bit
    s->set_saturation(s, -2); // lower the saturation
  }
  // drop down frame size for higher initial frame rate
  s->set_framesize(s, FRAMESIZE_QVGA);

#if defined(CAMERA_MODEL_M5STACK_WIDE) || defined(CAMERA_MODEL_M5STACK_ESP32CAM)
  s->set_vflip(s, 1);
  s->set_hmirror(s, 1);
#endif


  wm.setConfigPortalBlocking(false);
  wm.setConfigPortalTimeout(60);

  // reset settings - wipe stored credentials for testing
  // these are stored by the esp library
  //wm.resetSettings();

  // Automatically connect using saved credentials,
  // if connection fails, it starts an access point with the specified name ( "AutoConnectAP"),
  // if empty will auto generate SSID, if password is blank it will be anonymous AP (wm.autoConnect())
  // then goes into a blocking loop awaiting configuration and will return success result

  bool res;
  // res = wm.autoConnect(); // auto generated AP name from chipid
  // res = wm.autoConnect("AutoConnectAP"); // anonymous ap
  res = wm.autoConnect("AutoConnectAP", "password"); // password protected ap

  if (!res) {
    Serial.println("Failed to connect");
    freq = 550;
    ledcSetup(3, freq, something);
    ledcAttachPin(ledPin, 3);
    // increase the LED brightness

    for (int dutyCycle = 0; dutyCycle <= 255; dutyCycle++) {
      // changing the LED brightness with PWM
      ledcWrite(3, dutyCycle);
      delay(8);
    }

    for (int dutyCycle = 255; dutyCycle >= 0; dutyCycle--) {
      // changing the LED brightness with PWM
      ledcWrite(3, dutyCycle);
      delay(8);
    }
    wm.setConfigPortalBlocking(true);
    wm.setConfigPortalTimeout(60);
    wm.autoConnect("AutoConnectAP", "password"); // password protected ap
  }
  else {
    freq = 10000;
    ledcSetup(2, freq, something);
    ledcAttachPin(ledPin, 2);
    for (int i = 0; i < 3; i++) {
      for (int dutyCycle = 100; dutyCycle <= 250;  dutyCycle++) {
        // changing the LED brightness with PWM
        ledcWrite(2, dutyCycle);
        delay(2);
      }

      for (int dutyCycle = 250; dutyCycle >= 100; dutyCycle--) {
        // changing the LED brightness with PWM
        ledcWrite(2, dutyCycle);
        delay(2);
      }
    }

    //if you get here you have connected to the WiFi
    Serial.println("connected...yeey :)");
  }

  startCameraServer();

  Serial.print("Camera Ready! Use 'http://");
  Serial.print(WiFi.localIP());
  Serial.println("' to connect");
  if (!MDNS.begin("esp32")) {
    Serial.println("Error setting up MDNS responder!");
    while (1) {
      delay(1000);
    }
  }
  MDNS.addService("http", "tcp", 80);

}

void loop() {
  wm.process();
  if (HorPosVal != HorPos) {
    Serial.println(HorPos);
    uint8_t pos = panServo.read();
    if (HorPos - pos > 0) {
      for (uint8_t i = pos; i < HorPos; i++) {
        panServo.write(i);
        delay(5);
      }
    }
    if (HorPos - pos < 0) {
      for (uint8_t i = pos; i > HorPos; i--) {
        panServo.write(i);
        delay(5);
      }
    }
  }
  if (VertPosVal != VertPos) {
    uint8_t pos = tiltServo.read();
    if (VertPos - pos > 0) {
      for (uint8_t i = pos; i < VertPos; i++) {
        tiltServo.write(i);
        delay(5);
      }
    }
    if (VertPos - pos < 0) {
      for (uint8_t i = pos; i > VertPos; i--) {
        tiltServo.write(i);
        delay(5);
      }
    }
  }
  HorPosVal = HorPos;
  VertPosVal = VertPos;
  delay(5);
}

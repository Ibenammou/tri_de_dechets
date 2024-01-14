#include "camera_pins.h"
#include "camera_.h"
#include "esp_camera.h"
// Pin definition for CAMERA_MODEL_AI_THINKER
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
 
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// Setup WiFI SSID & Password
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// Set web server port number to 80
WiFiServer server(80);

// Variable to store HTTP request
String header;

// Motor Variables
//String motor_1_State = "off";
//String motor_2_State = "off";
String ledState = "off";

// GPIO Pins Variables
const int led = LED_BUILTIN;

// Setup the GPIO's and start the Web Server
void setup() {
  Serial.begin(115200);
  
  // Initialize GPIO's
  pinMode(led, OUTPUT);
  
  // Set Pins to Low
  digitalWrite(led, LOW);

  // Connect to WiFi
  Serial.print("Connecting to ");
  Serial.println(ssid);
  WiFi.begin(ssid,password);
  while(WiFi.status() != WL_CONNECTED){
    delay(500);
    Serial.print(".");
  }

  // Print local IP address and start web server
  Serial.println("");
  Serial.println("WiFi connected.");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());
  server.begin();
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
  // Frame parameters
  config.frame_size = FRAMESIZE_UXGA;
  config.jpeg_quality = 12;
  config.fb_count = 1;

  // Initialize the camera
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }

}

// Look for clients, connect to them and exchange data
void loop() {
  // Listen for Incoming Clients
  WiFiClient client = server.available();
  
  // If no client found, keep searching
  if (!client) {
    return;
  }

  // Wait until the client sends some data
  Serial.println("New client connected.");
  
  // Wait till Client sends some data
  while(!client.available()){
    delay(1);
  }

  // Read data from Client
  String clientResponse = client.readStringUntil('\r');
  Serial.println(clientResponse);
  client.flush();

  // Control the Motors
  // Default Values
  int motor1ControlVal = 101;
  int motor2ControlVal = 102;
  String motorNum = "";

  // Control Motor-1
  if (clientResponse.indexOf("/motor/1/1") != -1){
    // Motor On
    motor1ControlVal = 1;
    motorNum = "1";
  }
  else if (clientResponse.indexOf("/motor/1/0") != -1) {
    // Motor Off
    motor1ControlVal = 0;
    motorNum = "1";
  }
  // Control Motor-2
  else if (clientResponse.indexOf("/motor/2/1") != -1) {
    // Motor On
    motor2ControlVal = 1;
    motorNum = "2";
  }
  else if (clientResponse.indexOf("/motor/2/0") != -1) {
    // Motor Off
    motor2ControlVal = 0;
    motorNum = "2";
  }
  else {
    Serial.println("Invalid Request");
    client.stop();
    return;
  }

  // Debug
  Serial.println("......Recieved Command.....");
  Serial.println(clientResponse.indexOf("/motor/1"));
  Serial.println("Motor-1 Value: ");
  Serial.println(motor1ControlVal);
  Serial.println("Motor-2 Value: ");
  Serial.println(motor2ControlVal);
  Serial.println("...........................");
  
  // Return the response to Client
  client.println("HTTP/1.1 200 OK");
  client.println("Content-Type: text/html");
  client.println("");
  client.print("Controlling Motor Number: ");
  client.print(motorNum);
}

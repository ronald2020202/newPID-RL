#include <PID_v1.h>
#include <WiFiS3.h>
///////////////////////////////////////////////////////////////////////////// WORKING 2025-08-04 17:55 /////////////////////
///////////////////////////////////////////////////////////////////////////// Motor control over wifi works, now time to get RL ///////////

// WiFi credentials - CHANGE THESE TO YOUR NETWORK
const char* ssid = "TP-Link Li Home";
const char* password = "0MGSM62CL163";

// Server settings
WiFiServer server(80);
WiFiClient client;
bool clientConnected = false;

// Motor pins
const int MOTOR_PWM_PIN = 10;
const int MOTOR_DIR_PIN1 = 11;
const int MOTOR_DIR_PIN2 = 9;

// Encoder pins
const int ENCODER_A_PIN = 3;
const int ENCODER_B_PIN = 4;

// Encoder variables
volatile long encoderTicks = 0;
volatile bool lastEncoderA = false;

// Motor configuration (from your working code)
const int TICKS_PER_REVOLUTION = 1800;

// PID variables
double pidSetpoint = 0.0;
double pidInput = 0.0;
double pidOutput = 0.0;

// PID tuning parameters (from your working code)
double Kp = 10;
double Ki = 0;
double Kd = 30;

// Create PID controller
PID motorPID(&pidInput, &pidOutput, &pidSetpoint, Kp, Ki, Kd, DIRECT);

// Global variables (using your working approach)
bool PID_ENABLED = false;
double goalDegrees = 0.0;
static int stableCount = 0;

// PID calculation variables (from your working code)
static double lastError = 0;
static double integral = 0;

// Status reporting
unsigned long lastStatusTime = 0;
const unsigned long STATUS_INTERVAL = 100; // Send status every 100ms

void setup() {
  Serial.begin(115200);
  delay(2000);
  
  Serial.println("=== Arduino R4 WiFi PID Motor Control ===");
  
  // Setup pins (exactly like your working code)
  pinMode(MOTOR_PWM_PIN, OUTPUT);
  pinMode(MOTOR_DIR_PIN1, OUTPUT);
  pinMode(MOTOR_DIR_PIN2, OUTPUT);
  pinMode(ENCODER_A_PIN, INPUT_PULLUP);
  pinMode(ENCODER_B_PIN, INPUT_PULLUP);
  
  // Setup encoder interrupt (exactly like your working code)
  lastEncoderA = digitalRead(ENCODER_A_PIN);
  attachInterrupt(digitalPinToInterrupt(ENCODER_A_PIN), encoderISR, CHANGE);
  
  // Setup PID (exactly like your working code)
  motorPID.SetMode(AUTOMATIC);
  motorPID.SetOutputLimits(-255, 255);
  motorPID.SetSampleTime(10);
  
  // Start with motor stopped
  stopMotor();
  
  // Initialize WiFi
  setupWiFi();
  
  Serial.println("System ready!");
  printPIDValues();
}

void loop() {
  // Handle WiFi client connections
  handleWiFiClient();
  
  // Run PID control if enabled
  if (PID_ENABLED) {
    runPIDControl();
  }
  
  // Send periodic status updates to connected client
  sendStatusUpdate();
  
  delay(10);
}

void setupWiFi() {
  Serial.print("Connecting to WiFi network: ");
  Serial.println(ssid);
  
  // Start WiFi connection
  WiFi.begin(ssid, password);
  
  // Wait for connection with timeout
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(1000);
    Serial.print(".");
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println();
    Serial.println("WiFi connected successfully!");
    
    // Wait a moment for IP assignment and retry if needed
    delay(2000);
    IPAddress ip = WiFi.localIP();
    
    // If IP is still 0.0.0.0, try to refresh
    if (ip == IPAddress(0, 0, 0, 0)) {
      Serial.println("IP not assigned yet, waiting...");
      for (int i = 0; i < 10; i++) {
        delay(1000);
        ip = WiFi.localIP();
        if (ip != IPAddress(0, 0, 0, 0)) break;
      }
    }
    
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());
    Serial.print("MAC address: ");
    uint8_t mac[6];
    WiFi.macAddress(mac);
    char macStr[18];
    snprintf(macStr, sizeof(macStr), "%02X:%02X:%02X:%02X:%02X:%02X", 
            mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
    Serial.println(macStr);
        
    // Start the server
    server.begin();
    Serial.println("TCP server started on port 80");
    Serial.print("Connect to: http://");
    Serial.println(WiFi.localIP());
    Serial.println("Waiting for client connections...");
  } else {
    Serial.println();
    Serial.println("Failed to connect to WiFi!");
    Serial.println("Please check your credentials and try again.");
  }
}

void handleWiFiClient() {
  // Check for new client connections
  if (!clientConnected) {
    client = server.available();
    if (client) {
      clientConnected = true;
      Serial.println("Client connected!");
      sendMessage("CONNECTED Arduino R4 WiFi PID Controller Ready");
      sendMessage("STATUS " + getStatusString());
    }
  }
  
  // Handle existing client - FIXED: Check connection more carefully
  if (clientConnected) {
    if (client && client.connected()) {
      if (client.available()) {
        String command = client.readStringUntil('\n');
        command.trim();
        processWiFiCommand(command);
      }
    } else {
      // Client disconnected
      clientConnected = false;
      Serial.println("Client disconnected");
    }
  }
}

void processWiFiCommand(String command) {
  // Clean the command like your working serial code
  command.trim();
  command.replace("\r", "");
  command.replace("\n", "");
  
  Serial.print("Received: '");
  Serial.print(command);
  Serial.println("'");
  
  // Convert to uppercase for consistency
  command.toUpperCase();
  
  if (command == "FORWARD" || command == "F") {
    Serial.println("Forward command - disabling PID");
    PID_ENABLED = false;
    motorForward(15);  // Use same low speed as your working code
    sendMessage("OK Motor forward at speed 15");
    
  } else if (command == "REVERSE" || command == "R") {
    Serial.println("Reverse command - disabling PID");
    PID_ENABLED = false;
    motorReverse(15);  // Use same low speed as your working code
    sendMessage("OK Motor reverse at speed 15");
    
  } else if (command == "STOP" || command == "S") {
    Serial.println("Stop command");
    PID_ENABLED = false;
    stopMotor();
    sendMessage("OK Motor stopped");
    
  } else if (command == "STATUS") {
    sendMessage("STATUS " + getStatusString());
    
  } else if (command == "ZERO" || command == "Z") {
    zeroEncoder();
    sendMessage("OK Encoder zeroed");
    
  } else if (command == "PID_ON" || command == "P") {
    PID_ENABLED = true;
    sendMessage("OK PID enabled");
    
  } else if (command == "PID_OFF") {
    PID_ENABLED = false;
    stopMotor();
    sendMessage("OK PID disabled");
    
  } else if (command.startsWith("TARGET ")) {
    // Parse target in degrees (not ticks like before)
    double targetDeg = command.substring(7).toFloat();
    setTargetDegrees(targetDeg);
    sendMessage("OK Target set to " + String(targetDeg) + " degrees");
    
  } else if (command.startsWith("PID ")) {
    // Format: PID kp ki kd
    if (parsePIDCommand(command)) {
      sendMessage("OK PID parameters updated: Kp=" + String(Kp, 3) + 
                 " Ki=" + String(Ki, 3) + " Kd=" + String(Kd, 3));
    } else {
      sendMessage("ERROR Invalid PID format. Use: PID kp ki kd");
    }
    
  } else if (command == "0") {
    setTargetDegrees(0.0);
    sendMessage("OK Target set to 0 degrees");
    
  } else if (command == "1") {
    setTargetDegrees(90.0);
    sendMessage("OK Target set to 90 degrees");
    
  } else if (command == "2") {
    setTargetDegrees(180.0);
    sendMessage("OK Target set to 180 degrees");
    
  } else if (command == "3") {
    setTargetDegrees(270.0);
    sendMessage("OK Target set to 270 degrees");
    
  } else if (command == "HELP") {
    sendHelp();
    
  } else {
    sendMessage("ERROR Unknown command: " + command);
  }
}

bool parsePIDCommand(String command) {
  // Remove "PID " prefix
  command = command.substring(4);
  command.trim();
  
  // Parse three values
  int firstSpace = command.indexOf(' ');
  int secondSpace = command.indexOf(' ', firstSpace + 1);
  
  if (firstSpace == -1 || secondSpace == -1) {
    return false;
  }
  
  double newKp = command.substring(0, firstSpace).toDouble();
  double newKi = command.substring(firstSpace + 1, secondSpace).toDouble();
  double newKd = command.substring(secondSpace + 1).toDouble();
  
  // Validate values (basic sanity check)
  if (newKp < 0 || newKi < 0 || newKd < 0) {
    return false;
  }
  
  // Update PID parameters
  Kp = newKp;
  Ki = newKi;
  Kd = newKd;
  motorPID.SetTunings(Kp, Ki, Kd);
  
  return true;
}

void sendMessage(String message) {
  if (clientConnected && client && client.connected()) {
    client.println(message);
  }
  Serial.println("Sent: " + message);
}

void sendHelp() {
  sendMessage("HELP Available commands:");
  sendMessage("HELP FORWARD - Motor forward");
  sendMessage("HELP REVERSE - Motor reverse");
  sendMessage("HELP STOP - Stop motor");
  sendMessage("HELP TARGET <degrees> - Set PID target position");
  sendMessage("HELP PID <kp> <ki> <kd> - Set PID parameters");
  sendMessage("HELP PID_ON - Enable PID control");
  sendMessage("HELP PID_OFF - Disable PID control");
  sendMessage("HELP STATUS - Get current status");
  sendMessage("HELP ZERO - Zero encoder");
  sendMessage("HELP HELP - Show this help");
}

String getStatusString() {
  noInterrupts();
  long currentTicks = encoderTicks;
  interrupts();
  
  double degrees = ticksToDegrees(currentTicks);
  double error = 0;
  
  if (PID_ENABLED) {
    error = shortestAngularDistance(degrees, goalDegrees);
  }
  
  String status = "";
  status += "ticks:" + String(currentTicks);
  status += ",degrees:" + String(degrees, 1);
  status += ",target_degrees:" + String(goalDegrees, 1);
  status += ",error:" + String(error, 1);
  status += ",pid_enabled:" + String(PID_ENABLED ? "true" : "false");
  status += ",kp:" + String(Kp, 3);
  status += ",ki:" + String(Ki, 3);
  status += ",kd:" + String(Kd, 3);
  status += ",stable_count:" + String(stableCount);
  status += ",wifi_connected:" + String(WiFi.status() == WL_CONNECTED ? "true" : "false");
  status += ",client_connected:" + String(clientConnected ? "true" : "false");
  
  return status;
}

void sendStatusUpdate() {
  if (clientConnected && client && client.connected()) {
    unsigned long currentTime = millis();
    if (currentTime - lastStatusTime >= STATUS_INTERVAL) {
      sendMessage("STATUS " + getStatusString());
      lastStatusTime = currentTime;
    }
  }
}

void printPIDValues() {
  Serial.print("PID: Kp=");
  Serial.print(Kp, 3);
  Serial.print(" Ki=");
  Serial.print(Ki, 3);
  Serial.print(" Kd=");
  Serial.println(Kd, 3);
}

// Convert encoder ticks to degrees (from your working code)
double ticksToDegrees(long ticks) {
  double degrees = (double(ticks) * 360.0) / TICKS_PER_REVOLUTION;
  while (degrees < 0) degrees += 360.0;
  while (degrees >= 360.0) degrees -= 360.0;
  return degrees;
}

// Calculate shortest angular distance (from your working code)
double shortestAngularDistance(double current, double target) {
  double diff = target - current;
  
  // Normalize difference to [-180, 180]
  while (diff > 180.0) diff -= 360.0;
  while (diff < -180.0) diff += 360.0;
  
  // Special case: if difference is exactly ±180, choose positive direction
  if (abs(diff) == 180.0) {
    diff = 180.0;
  }
  
  return -diff;
}

void setTargetDegrees(double targetDegrees) {
  while (targetDegrees < 0) targetDegrees += 360.0;
  while (targetDegrees >= 360.0) targetDegrees -= 360.0;
  
  goalDegrees = targetDegrees;
  PID_ENABLED = true;
  stableCount = 0;
  
  // Reset PID state
  lastError = 0;
  integral = 0;
  
  Serial.print("Target set to: ");
  Serial.print(targetDegrees, 1);
  Serial.println("°");
}

// Use your working PID control logic
void runPIDControl() {
  // Get current position safely
  noInterrupts();
  long currentTicks = encoderTicks;
  interrupts();
  
  double currentDegrees = ticksToDegrees(currentTicks);
  double angularError = shortestAngularDistance(currentDegrees, goalDegrees);
  
  // Stability check (from your working code)
  const int REQUIRED_STABLE_COUNT = 50;
  const double ERROR_THRESHOLD = 1.0;
  
  if (abs(angularError) <= ERROR_THRESHOLD) {
    stableCount++;
    if (stableCount >= REQUIRED_STABLE_COUNT) {
      stopMotor();
      static unsigned long lastStoppedPrint = 0;
      if (millis() - lastStoppedPrint >= 2000) {
        String msg = "STABLE Target reached. Error: " + String(angularError, 1) + "°";
        sendMessage(msg);
        Serial.println("Motor stopped - target reached");
        lastStoppedPrint = millis();
      }
      return;
    }
  } else {
    stableCount = 0;
  }
  
  // Manual PID calculation (exactly from your working code)
  double derivative = angularError - lastError;
  lastError = angularError;
  
  // Calculate integral (with windup protection)
  integral += angularError;
  integral = constrain(integral, -100, 100);
  
  // Manual PID calculation
  double pidControlOutput = (Kp * angularError) + (Ki * integral) + (Kd * derivative);
  
  // The output should be negative to correct positive error
  double finalSpeed = -pidControlOutput;
  
  // Apply minimum speed (from your working code)
  if (abs(finalSpeed) > 0 && abs(finalSpeed) < 15) {
    finalSpeed = (finalSpeed > 0) ? 15 : -15;
  }
  
  // Apply maximum speed limit
  if (abs(finalSpeed) > 255) {
    finalSpeed = (finalSpeed > 0) ? 255 : -255;
  }
  
  // Apply to motor (exactly like your working code)
  if (abs(finalSpeed) < 2) {
    stopMotor();
  } else if (finalSpeed > 0) {
    motorForward((int)abs(finalSpeed));
  } else {
    motorReverse((int)abs(finalSpeed));
  }
  
  // Debug output (from your working code)
  static unsigned long lastPrint = 0;
  if (millis() - lastPrint >= 100) {
    Serial.print("T:");
    Serial.print(goalDegrees, 0);
    Serial.print(" C:");
    Serial.print(currentDegrees, 0);
    Serial.print(" E:");
    Serial.print(angularError, 1);
    Serial.print(" M:");
    Serial.println((int)finalSpeed);
    lastPrint = millis();
  }
}

void encoderISR() {
  static unsigned long lastInterruptTime = 0;
  unsigned long currentTime = micros();
  
  if (currentTime - lastInterruptTime < 200) {
    return;
  }
  lastInterruptTime = currentTime;
  
  bool currentA = digitalRead(ENCODER_A_PIN);
  bool currentB = digitalRead(ENCODER_B_PIN);
  
  if (currentA != lastEncoderA) {
    if (currentA == currentB) {
      encoderTicks--;
    } else {
      encoderTicks++;
    }
  }
  
  lastEncoderA = currentA;
}

void motorForward(int speed) {
  speed = constrain(speed, 0, 255);
  digitalWrite(MOTOR_DIR_PIN1, HIGH);
  digitalWrite(MOTOR_DIR_PIN2, LOW);
  analogWrite(MOTOR_PWM_PIN, speed);
}

void motorReverse(int speed) {
  speed = constrain(speed, 0, 255);
  digitalWrite(MOTOR_DIR_PIN1, LOW);
  digitalWrite(MOTOR_DIR_PIN2, HIGH);
  analogWrite(MOTOR_PWM_PIN, speed);
}

void stopMotor() {
  digitalWrite(MOTOR_DIR_PIN1, HIGH);
  digitalWrite(MOTOR_DIR_PIN2, HIGH);
  analogWrite(MOTOR_PWM_PIN, 0);
}

void zeroEncoder() {
  noInterrupts();
  encoderTicks = 0;
  interrupts();
}
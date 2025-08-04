#include <WiFiS3.h>

// WiFi credentials - CHANGE THESE TO YOUR NETWORK
const char* ssid = "TP-Link Li Home";
const char* password = "0MGSM62CL163";

// Server settings
WiFiServer server(80);
WiFiClient client;

void setup() {
  Serial.begin(115200);
  delay(2000);
  
  Serial.println("=== Basic Arduino WiFi Test ===");
  
  // Connect to WiFi
  Serial.print("Connecting to WiFi: ");
  Serial.println(ssid);
  
  WiFi.begin(ssid, password);
  
  // Wait for connection
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.print(".");
  }
  
  Serial.println();
  Serial.println("WiFi connected!");
  
  // Wait a bit for IP assignment
  delay(2000);
  
  // Display connection info
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
  Serial.print("MAC address: ");
  uint8_t mac[6];
    WiFi.macAddress(mac);
    char macStr[18];
    snprintf(macStr, sizeof(macStr), "%02X:%02X:%02X:%02X:%02X:%02X", 
            mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
    Serial.println(macStr);
    
  
  // Start server
  server.begin();
  Serial.println("Server started on port 80");
  Serial.print("Test URL: http://");
  Serial.println(WiFi.localIP());
  Serial.println("Ready for connections!");
}

void loop() {
  // Check for new clients
  client = server.available();
  
  if (client) {
    Serial.println("New client connected");
    
    // Read the request
    String request = "";
    while (client.connected() && client.available()) {
      char c = client.read();
      request += c;
      
      // If we hit the end of the line, we're done reading
      if (c == '\n') {
        break;
      }
    }
    
    Serial.print("Request: ");
    Serial.println(request.substring(0, 50)); // Print first 50 chars
    
    // Send response
    client.println("HTTP/1.1 200 OK");
    client.println("Content-Type: text/html");
    client.println("Connection: close");
    client.println();
    client.println("<!DOCTYPE html>");
    client.println("<html>");
    client.println("<head><title>Arduino Test</title></head>");
    client.println("<body>");
    client.println("<h1>Arduino WiFi Test</h1>");
    client.println("<p>Connection successful!</p>");
    client.println("<p>Arduino IP: " + WiFi.localIP().toString() + "</p>");
    client.println("<p>Time: " + String(millis()) + " ms</p>");
    client.println("</body>");
    client.println("</html>");
    
    // Close connection
    client.stop();
    Serial.println("Client disconnected");
  }
  
  delay(10);
}
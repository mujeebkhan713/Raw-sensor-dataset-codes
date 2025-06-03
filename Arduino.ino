#include <ESP8266WiFi.h>
#include <ThingSpeak.h>
#include <DHT.h>

#define DHTPIN D4 // Define the pin where the DHT11 is connected
#define DHTTYPE DHT11
#define RELAY_PIN D1 // Define the pin where the relay is connected

const char* ssid = "MAKIphone";
const char* password = "mujeebkhan777";
const char* server = "api.thingspeak.com";
unsigned long myChannelNumber = 2571096;
const char* myWriteAPIKey = "MZ4SZ89H0GLFBZXO";
const char* myReadAPIKey = "J5TF4QOAPUQOV44R";

WiFiClient client;
DHT dht(DHTPIN, DHTTYPE);

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("Connected to WiFi");
  ThingSpeak.begin(client);
  dht.begin();

  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, LOW); // Relay off by default
}

void loop() {
  float h = dht.readHumidity();
  float t = dht.readTemperature();
  int smoke = analogRead(A0);

  Serial.print("Temperature: ");
  Serial.print(t);
  Serial.print(" *C, Humidity: ");
  Serial.print(h);
  Serial.print(" %, Smoke: ");
  Serial.println(smoke);

  if (!isnan(h) && !isnan(t)) { // Check if the readings are valid
    ThingSpeak.setField(1, t);
    ThingSpeak.setField(2, h);
    ThingSpeak.setField(3, smoke);

    int response = ThingSpeak.writeFields(myChannelNumber, myWriteAPIKey);
    if(response == 200) {
      Serial.println("Channel update successful.");
    } else {
      Serial.println("Problem updating channel. HTTP error code " + String(response));
    }
  } else {
    Serial.println("Failed to read from DHT sensor!");
  }

  // Read the prediction result from ThingSpeak
  int firePrediction = ThingSpeak.readLongField(myChannelNumber, 4, myReadAPIKey); // Assuming field 4 is used for fire prediction
  if (firePrediction == 1) {
    digitalWrite(RELAY_PIN, HIGH); // Turn on relay
  } else {
    digitalWrite(RELAY_PIN, LOW); // Turn off relay
  }

  delay(20000); // Wait for 15 seconds to meet ThingSpeak rate limit
}

#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>
#include <MPU6050.h>

MPU6050 mpu;

const char* ssid = "Etisalat 4G iModem-DAAB";
const char* password = "12061884";


WiFiUDP udp;

IPAddress broadcastIP(255,255,255,255);

const int pcPort = 5005;

String HAND_NAME = "LEFT";

const int flexPins[5] = {36,33,32,35,34};

void setup() {

  Serial.begin(115200);

  Wire.begin(21,22);

  mpu.initialize();

  WiFi.begin(ssid,password);

  while(WiFi.status()!=WL_CONNECTED){

    delay(500);
    Serial.print(".");
  }

  Serial.println("\nWiFi Connected");

  udp.begin(pcPort);
}

void loop() {

  int16_t ax, ay, az, gx, gy, gz;

  mpu.getMotion6(&ax,&ay,&az,&gx,&gy,&gz);

  int flex[5];

  for(int i=0;i<5;i++){

    flex[i]=analogRead(flexPins[i]);
  }

  char msg[128];

  sprintf(
    msg,
    "%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d",

    HAND_NAME.c_str(),

    flex[0],
    flex[1],
    flex[2],
    flex[3],
    flex[4],

    ax,
    ay,
    az,

    gx,
    gy,
    gz
  );

  udp.beginPacket(
    broadcastIP,
    pcPort
  );

  udp.write(
    (uint8_t*)msg,
    strlen(msg)
  );

  udp.endPacket();

  delay(2);
}
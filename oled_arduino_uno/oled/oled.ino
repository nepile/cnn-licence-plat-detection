#include "arduino_secrets.h"

#include "U8glib.h"

U8GLIB_SSD1306_128X64 u8g(U8G_I2C_OPT_DEV_0 | U8G_I2C_OPT_NO_ACK | U8G_I2C_OPT_FAST);

char receivedText[32] = "Menunggu Koneksi";
String inputString = "";
bool receiving = false;

unsigned long lastReceiveTime = 0;
const unsigned long timeoutDuration = 5000; 
unsigned long lastDisplay = 0;
bool dataBaruDiterima = false;

void draw(void) {
  u8g.setFont(u8g_font_unifont);
  u8g.drawStr(0, 22, receivedText);
}

void setup(void) {
  Serial.begin(9600);
  Serial.setTimeout(100);

  if (u8g.getMode() == U8G_MODE_R3G3B2) {
    u8g.setColorIndex(255);
  } else if (u8g.getMode() == U8G_MODE_GRAY2BIT) {
    u8g.setColorIndex(3);
  } else if (u8g.getMode() == U8G_MODE_BW) {
    u8g.setColorIndex(1);
  } else if (u8g.getMode() == U8G_MODE_HICOLOR) {
    u8g.setHiColorByRGB(255, 255, 255);
  }

  pinMode(8, OUTPUT);
  lastReceiveTime = millis();
}

void loop(void) {
  // Baca Serial
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      receiving = false;
      inputString.trim(); 
      inputString.toCharArray(receivedText, sizeof(receivedText));
      Serial.print("Diterima: ");
      Serial.println(receivedText);
      lastReceiveTime = millis();
      dataBaruDiterima = true;
      inputString = "";
    } else {
      inputString += c;
      receiving = true;
    }
  }

  if (millis() - lastDisplay > 100) {
    u8g.firstPage();
    do {
      draw();
    } while (u8g.nextPage());
    lastDisplay = millis();
  }

  if (millis() - lastReceiveTime > timeoutDuration && dataBaruDiterima) {
    Serial.println("Timeout: Kembali ke 'Menunggu...'");
    strcpy(receivedText, "Menunggu...");
    dataBaruDiterima = false;
  }

  delay(2000); 
}

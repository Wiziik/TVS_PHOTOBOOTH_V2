/*
  Arduino Leonardo USB-MIDI toggle switch
  - Uses MIDIUSB library
  - Toggle switch wired to GND + digital pin using INPUT_PULLUP
  - Sends MIDI Note On when switch is ON, Note Off when OFF (on state change only)
  - Non-blocking debounce
*/

#include <MIDIUSB.h>

// ========= Config =========
const uint8_t SWITCH_PIN   = 2;    // Digital pin connected to toggle switch
const uint8_t MIDI_CH      = 1;    // 1..16
const uint8_t NOTE_NUMBER  = 60;   // Middle C = 60
const uint8_t VELOCITY_ON  = 100;  // 1..127
const uint8_t VELOCITY_OFF = 0;    // usually 0 for Note Off
const unsigned long DEBOUNCE_MS = 15;
// ==========================

// If you wire the toggle so that ON connects the pin to GND:
// - INPUT_PULLUP => pin reads LOW when ON, HIGH when OFF.
const bool ON_IS_LOW = true;

static bool stableState = false;          // false=OFF, true=ON (our logical meaning)
static bool lastRawLogical = false;
static unsigned long lastChangeMs = 0;

static inline uint8_t midiChannelNibble(uint8_t ch1to16) {
  // USB-MIDI "status" low nibble is channel 0..15
  return (uint8_t)((ch1to16 - 1) & 0x0F);
}

void sendNoteOn(uint8_t note, uint8_t velocity, uint8_t ch) {
  midiEventPacket_t p = {0x09, (uint8_t)(0x90 | midiChannelNibble(ch)), note, velocity};
  MidiUSB.sendMIDI(p);
  MidiUSB.flush();
}

void sendNoteOff(uint8_t note, uint8_t velocity, uint8_t ch) {
  // You can use real Note Off (0x80) or Note On with vel=0. We'll use 0x80.
  midiEventPacket_t p = {0x08, (uint8_t)(0x80 | midiChannelNibble(ch)), note, velocity};
  MidiUSB.sendMIDI(p);
  MidiUSB.flush();
}

bool readSwitchLogical() {
  // rawRead is HIGH when open, LOW when connected to GND (with INPUT_PULLUP)
  int raw = digitalRead(SWITCH_PIN);
  bool isLow = (raw == LOW);
  bool logicalOn = ON_IS_LOW ? isLow : !isLow;
  return logicalOn;
}

void setup() {
  pinMode(SWITCH_PIN, INPUT_PULLUP);

  // Initialize stable state from current physical position
  stableState = readSwitchLogical();
  lastRawLogical = stableState;
  lastChangeMs = millis();

  // Optional: on boot, you can choose to send the current state once.
  // Comment out if you don't want any MIDI sent at startup.
  if (stableState) sendNoteOn(NOTE_NUMBER, VELOCITY_ON, MIDI_CH);
  else            sendNoteOff(NOTE_NUMBER, VELOCITY_OFF, MIDI_CH);
}

void loop() {
  const unsigned long now = millis();

  // Read current logical position
  bool rawLogical = readSwitchLogical();

  // Track last time raw changed (for debouncing)
  if (rawLogical != lastRawLogical) {
    lastRawLogical = rawLogical;
    lastChangeMs = now;
  }

  // If raw has been stable long enough and differs from stableState, commit it
  if ((now - lastChangeMs) >= DEBOUNCE_MS && rawLogical != stableState) {
    stableState = rawLogical;

    // Send MIDI only on stable state change
    if (stableState) {
      sendNoteOn(NOTE_NUMBER, VELOCITY_ON, MIDI_CH);
    } else {
      sendNoteOff(NOTE_NUMBER, VELOCITY_OFF, MIDI_CH);
    }
  }

  // No delay(); keep loop responsive
}

#include <Wire.h>  // Include Wire if you're using I2C
#include <SFE_MicroOLED.h>  // Include the SFE_MicroOLED library
#include <SparkFunBME280.h>
#include <SparkFunCCS811.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/experimental/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "Model.h"
#include "Database.h"
#define GPS_BAUD 115200
#define N_FLOATS 4


#define PIN_RESET 9  
#define DC_JUMPER 1 
#define ROTARY_ANGLE_SENSOR A16
#define ADC_REF 3.3 //reference voltage of ADC is 5v.If the Vcc switch on the seeeduino
                    //board switches to 3V3, the ADC_REF should be 3.3
#define GROVE_VCC 3.3 //VCC of the grove interface is normally 5v
#define FULL_ANGLE 300 //full value of the rotary angle is 300 degrees
#define CCS811_ADDR 0x5B //Default I2C Address

CCS811 myCCS811(CCS811_ADDR);
BME280 myBME280;
MicroOLED oled(PIN_RESET, DC_JUMPER);    // I2C declaration


// Globals, used for compatibility with Arduino-style sketches.
namespace { 
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
  // Create an area of memory to use for input, output, and intermediate arrays.
  constexpr int kTensorArenaSize = 21 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
} 

Ublox M8_Gps;
// Altitude - Latitude - Longitude - N Satellites
float gpsArray[N_FLOATS] = {0, 0, 0, 0};

void setup() {
   Serial.begin(SERIAL_BAUD);
   Serial1.begin(GPS_BAUD);
 
}

const int buttonPin = 2;     // the number of the pushbutton pin
int state = 0;
int buttonState = 0;

float longitude, latitude ;
int x , y;
float burned_area = 0; 
int wind_speed = 0;

void setup(void) 
{
  Serial.begin(9600);
  pinMode(ROTARY_ANGLE_SENSOR, INPUT);
  pinMode(buttonPin, INPUT);

  delay(100);
  Wire.begin();
  
  oled.begin();    // Initialize the OLED
  oled.clear(ALL); // Clear the display's internal memory
  oled.display();  // Display what's in the buffer (splashscreen)
  delay(1000);     // Delay 1000 ms
  oled.clear(PAGE); // Clear the buffer.

  //This begins the CCS811 sensor and prints error status of .begin()
  CCS811Core::status returnCode = myCCS811.begin();
  if (returnCode != CCS811Core::SENSOR_SUCCESS)
  {
    Serial.println("Problem with CCS811");
    printDriverError(returnCode);
  }
  else
  {
    Serial.println("CCS811 online");
  }

  //Initialize BME280
  //For I2C, enable the following and disable the SPI section
  myBME280.settings.commInterface = I2C_MODE;
  myBME280.settings.I2CAddress = 0x77;
  myBME280.settings.runMode = 3; //Normal mode
  myBME280.settings.tStandby = 0;
  myBME280.settings.filter = 4;
  myBME280.settings.tempOverSample = 5;
  myBME280.settings.pressOverSample = 5;
  myBME280.settings.humidOverSample = 5;

  //Calling .begin() causes the settings to be loaded
  delay(10);  //Make sure sensor had enough time to turn on. BME280 requires 2ms to start up.
  byte id = myBME280.begin(); //Returns ID of 0x60 if successful
  if (id != 0x60)
  {
    Serial.println("Problem with BME280");
  }
  else
  {
    Serial.println("BME280 online");
  }

  static tflite::ops::micro::AllOpsResolver resolver;
  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(   model,
                                                        resolver,
                                                        tensor_arena,
                                                        kTensorArenaSize,
                                                        error_reporter);
  interpreter = &static_interpreter;
  //Assign Memory to tensor Area
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  //Set Input and Output variable
  input = interpreter->input(0);
  output = interpreter->output(0);
}
 
void loop(void) 
{
   buttonState = digitalRead(buttonPin);
   int sensor_value = analogRead(ROTARY_ANGLE_SENSOR);
   switch(state)
   {
    case 0: state0(); break;
    case 1: state1(); break;
    case 2: state2(); break;
    case 3: state3(); break;
    case 4: state4(); break;
   }
  

  if (myCCS811.dataAvailable())
  {
    //Calling this function updates the global tVOC and eCO2 variables
    myCCS811.readAlgorithmResults();
    float BMEtempC = myBME280.readTempC();
    float BMEhumid = myBME280.readFloatHumidity();

    myCCS811.setEnvironmentalData(BMEhumid, BMEtempC);
  }
  delay(1000);
}


void state0()
{
  buttonState = digitalRead(buttonPin);
  int sensor_value = analogRead(ROTARY_ANGLE_SENSOR);
  if(sensor_value < 500)
  {
    oled.clear(PAGE); // Clear the buffer.
    oled.setFontType(0); // set font type 0, please see declaration in SFE_MicroOLED.cpp
    oled.setCursor(15, 10);
    oled.print("Weather     Data");
    oled.display(); 
    if (buttonState == LOW) {state = 1;}
  }
  else
  {
    oled.clear(PAGE); // Clear the buffer.
    oled.setFontType(0); // set font type 0, please see declaration in SFE_MicroOLED.cpp
    oled.setCursor(3, 10);
    oled.print("Prediction");
    oled.display(); 
    if (buttonState == LOW) {
      Serial.println("pressed");  
      state = 2 ;
      
    }
  }
}

void state1()
{
  buttonState = digitalRead(buttonPin);
  int sensor_value = analogRead(ROTARY_ANGLE_SENSOR);
  if(sensor_value < 200)
  {
    oled.clear(PAGE); // Clear the buffer.
    oled.setCursor(15, 12);
    oled.print("Temp ");
    
    oled.setCursor(15, 30);
    oled.print(myBME280.readTempF());
    oled.print(" F");
    oled.display();
  }
  else if (sensor_value >= 200 && sensor_value  < 400)
  {
    oled.clear(PAGE); // Clear the buffer.
    oled.setCursor(3, 12);
    oled.print("Humidity ");
    oled.setCursor(15, 30);
    oled.print(myBME280.readFloatHumidity());
    oled.display();
  }
  else if (sensor_value >= 400 && sensor_value  < 600)
  {
    oled.clear(PAGE); // Clear the buffer.
    oled.setCursor(6, 12);
    oled.print("Pressure ");
    oled.setCursor(3, 30);
    oled.print(myBME280.readFloatPressure());
    oled.display();
  }
  else if (sensor_value >= 600 && sensor_value  < 800)
  {
    oled.clear(PAGE); // Clear the buffer.
    oled.setCursor(15, 12);
    oled.print("CC02 ");
    oled.setCursor(15, 30);
    oled.print(myCCS811.getCO2());
    oled.display();
  }
  else 
  {
    oled.clear(PAGE); // Clear the buffer.
    oled.setCursor(15, 12);
    oled.print("TVOC ");
    oled.setCursor(15, 30);
    oled.print(myCCS811.getTVOC());
    oled.display();
  }
  if (buttonState == LOW) {state = 0 ;}
}

void state2()
{
    buttonState = digitalRead(buttonPin);
    wind_speed = analogRead(ROTARY_ANGLE_SENSOR);
    oled.clear(PAGE); // Clear the buffer.
    oled.setFontType(0); // set font type 0, please see declaration in SFE_MicroOLED.cpp
    oled.setCursor(3, 10);
    oled.print("Wind Speed");
    oled.setCursor(15, 40);
    oled.print(wind_speed);
    oled.display(); 
    if (buttonState == LOW) {state = 3 ;}
}

void state3()
{
  buttonState = digitalRead(buttonPin);
  int sensor_value = analogRead(ROTARY_ANGLE_SENSOR);
  oled.clear(PAGE); // Clear the buffer.
  oled.setFontType(0); // set font type 0, please see declaration in SFE_MicroOLED.cpp
  oled.setCursor(15, 10);
  oled.print("Start    Predition");
  oled.display(); 
  if (buttonState == LOW) {state = 4;}

}

void state4()
{
  buttonState = digitalRead(buttonPin);

  RunML();
  Serial.println("state 4");
  oled.clear(PAGE); // Clear the buffer.
  oled.setFontType(0);
  oled.setCursor(0, 10);
  oled.print("  Burned     Area ");
  oled.setCursor(15, 30);
  oled.print(burned_area);
  oled.display(); 

  //Display prediction
  if (buttonState == LOW) {state = 0; counter++; }
}


void RunML()
{
  //Read in local FFMC value (x = longitude, y= lattitude)
  input->data.f[0] = FFMC[x][y];
  //Read in local DMC value (x = longitude, y= lattitude)
  input->data.f[1] = DMC[x][y];
  //Read in local DC value (x = longitude, y= lattitude)
  input->data.f[2] = DC[x][y];
  //Read in local ISI value (x = longitude, y= lattitude)
  input->data.f[3] = ISI[x][y];
  //Read in local temp value
  input->data.f[4] = myBME280.readTempF();
  //Read in local humidity value
  input->data.f[5] = myBME280.readFloatHumidity();
  //Read in local wind value
  input->data.f[6] = wind_speed;
  // Run the model on the spectrogram input and make sure it succeeds.
  TfLiteStatus invoke_status = interpreter->Invoke();
      if (invoke_status != kTfLiteOk) {
      error_reporter->Report("Invoke failed");
      return;
  }    
  burned area = output->data.f[0]; 
}

void getGPS()
{
  latitude = M8_Gps.latitude;
  longitude = M8_Gps.longitude; 
  if( (longitude > 32 && longitude > 41) && ( latitude > -124 && latitude < -115) )
  {
    //round to nearest .5 
    longitude = (floor((longitude*2)+0.5)/2);
    latitude  = (floor((latitude*2)+0.5)/2);
    //
    x = longitude - 32;
    y = latitude  + 124; 
  {
  else
  {
    //outside valid coordinate
    Serial.println("Outside valid coordinate");
    return;
  {
}

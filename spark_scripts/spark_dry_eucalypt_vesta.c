// Dry Eucalypt model - Project Vesta
// modified geoff.goldrick@protonmail.com 2020-08-18

// -------------------------------------------
// Dry Eucalypt model - Project Vesta
// Model parameters
// These must be defined below, or included as a user-defined layer
//
// 1. Temperature, 'temp' (input)
// 2. Relative humidity, 'rel_hum' (input)
// 3. Surface fuel hazard score, 'fhs_surface'
// 4. Near-surface fuel hazard score, 'fhs_near_surface'
// 5. Near surface height, 'h_near_surface'
// -------------------------------------------

// Backing and flanking coefficients compared to head fire ROS
REAL wind_speed = length(wind_vector);
REAL wdot = dot(normalize(wind_vector),advect_normal_vector);

// Calculate length-to-breadth ratio (LBR)
REAL LBR = 1.0;
if (wind_speed < 5){
    LBR = 1.0;
} else if (wind_speed < 25){
    LBR = 0.9286 * exp(0.0505 * wind_speed); 
} else {
    LBR = 0.1143 * wind_speed + 0.4143; 
} 

// Determine coefficient for flank rank of spread, Rf = cf * Rh, using Andrew's LBR equations 
REAL cc = sqrt(1.0-pow(LBR, -2.0)); 
REAL cb = (1.0-cc)/(1.0+cc); 
REAL a_LBR = 0.5*(cb+1.0);
REAL cf = a_LBR/LBR; 

// Determine shape parameters 
REAL f = 0.5*(1.0+cb); 
REAL g = 0.5*(1.0-cb); 
REAL h = cf; 

// Now calculate a speed coefficient using normal flow formula 
REAL speed_fraction = (g*wdot+sqrt(h*h+(f*f-h*h)*wdot*wdot)); 

// Estimating fuel moisture content using Gould et al. (2007) and Matthews et al. (2010) 
// Initialising the fuel moisture variable 
REAL Mf; 
// Calculating fuel moisture between 12:00 and 16:59 (valid for sunny days from October to March) 
if (hour > 11 && hour < 17){
    Mf = 2.76 + (0.124*rel_hum) - (0.0187*temp);}

// Calculating fuel moisture for other daylight hours (from 9:00 to 11:59 and 17:00 to 19:59 in this example)
else if ((hour < 12 && hour > 8) || (hour > 16 & hour < 20)){
    Mf = 3.6 + (0.169*rel_hum) - (0.045*temp);} 

// Calculating fuel moisture for night time hours (from 20:00 to 8:59) in this example 
else{
    Mf = 3.08 + (0.198*rel_hum) - (0.0483*temp);} 

// Calculate moisture coefficients from Burrows (1999) 
REAL moisture_coeff = 18.35 * pow(Mf,-1.495); 

// Calculate spread rate (in m/s) 
REAL head_speed; 

if ( wind_speed > 5 )
  head_speed = (30.0 + 1.5308 * pow(wind_speed-5,0.8576) * 
           pow(fhs_surface,0.9301) * pow(fhs_near_surface*h_near_surface,0.6366) * 1.03 ) *
           moisture_coeff/3600;
else
  head_speed = 30.0 * moisture_coeff/3600;

// Adjust for calculated speed coefficient for fire flanks
speed = head_speed * speed_fraction;
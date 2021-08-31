// Dry Eucalypt model - McArthur

// -------------------------------------------
// Model parameters
// These must be defined below, or included as a user-defined layer
//
// 1. Temperature, 'temp' (input)
// 2. Relative humidity, 'rel_hum' (input)
// 3. Drought factor, 'D'
// 4. Fuel load, 'fuel_load' (input)
// -------------------------------------------

// Calculating FFDI using McArthur and Noble et al. (1980)
REAL FFDI = 2 * exp(-0.450 + 0.987*log(D) - 0.0345*rel_hum + 0.0338*temp + 0.0234*wind);

// Calculate spread rate in km/h using McArthur's Mk 5 and Noble et al. (1980)
speed = 0.0012 * FFDI * fuel_load;

// Converting spread rate into m/s
speed = speed / 3.6;
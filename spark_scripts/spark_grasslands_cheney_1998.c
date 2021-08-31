// lnc_fc_woodlands

// CSIRO grasslands models - Cheney et al. (1998)
// Subclasses defined as the following sub type of grasslands:
// 1 - Eaten out
// 2 - Cut / grazed
// 3 - Natural / undisturbed
// 4 - Woodland
// 5 - Open forest

// -------------------------------------------
// Model parameters
// 1. Temperature, 'temp'  (input)
// 2. Relative humidity, 'rel_hum' (input)
// 3. Curing value, 'curing'
// -------------------------------------------

// Calculating the wind speed which is used to calculate head fire ROS
REAL wind_speed = length(wind_vector);

// Calculating the normalised dot product between the wind vector and the normal to the fire perimeter
REAL wdot = dot(normalize(wind_vector),advect_normal_vector);

// Calculate length-to-breadth ratio (LBR) which varies with wind speed
// Equations are curve fits adapted from Taylor (1997)
REAL LBR = 1.0;
if (wind_speed < 5){
  LBR = 1.0;
} else {
  LBR = 1.1*pow(wind_speed, 0.464);
}

// Determine coefficient for backing and flanking rank of spread using elliptical equations
// Where R_backing = cb * R_head, R_flanking = cf * R_head,
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

// Calculate curing coefficient from Cruz et al. (2015)
REAL curing_coeff;
if ( curing < 20 )
    curing_coeff = 0;
else
    curing_coeff = 1.036 / ( 1 + 103.989 * exp( -0.0996 * (curing - 20) ) );

// Fuel moisture content approximated using McArthur (1966)
REAL GMf = 9.58-(0.205*temp) + (0.138*rel_hum);

// Calculate moisture coefficient from Cheney et al. (1998)
REAL moisture_coeff;
if ( GMf <= 12 ){
    moisture_coeff = exp( -0.108 * GMf );
} else if ( wind_speed <= 10 ){
    moisture_coeff = 0.684 - 0.0342 * GMf; 
} else{
    moisture_coeff = 0.547 - 0.0228 * GMf;} 

// Defining coefficients for the various grasslands sub-types using different subclasses.
REAL head_speed; 
REAL CF_Backing_Slow; 
REAL CF_Backing_Fast; 
REAL CF_Wind_Slow; 
REAL CF_Wind_Fast; 
REAL speed_factor = 1; 

if (subclass == 1) // Eaten out 
    {
    CF_Backing_Slow = 0.027;
    CF_Backing_Fast = 0.55;
    CF_Wind_Slow = 0.1045; 
    CF_Wind_Fast = 0.3575;
    } 
else if (subclass == 2) // Cut or grazed 
    {
    CF_Backing_Slow = 0.054;
    CF_Backing_Fast = 1.1;
    CF_Wind_Slow = 0.209;
    CF_Wind_Fast = 0.715; 
    } 
else if (subclass == 3) // Natural or undisturbed
    {
    CF_Backing_Slow = 0.054;
    CF_Backing_Fast = 1.4;
    CF_Wind_Slow = 0.269;
    CF_Wind_Fast = 0.838;
    } 
else if (subclass == 4) // Woodland  
    {
    CF_Backing_Slow = 0.054;
    CF_Backing_Fast = 1.4;
    CF_Wind_Slow = 0.269;
    CF_Wind_Fast = 0.838;
    speed_factor = 0.5; 
    } 
else if (subclass == 5) // Open forest
    {
    CF_Backing_Slow = 0.054;
    CF_Backing_Fast = 1.4;
    CF_Wind_Slow = 0.269;
    CF_Wind_Fast = 0.838;
    speed_factor = 0.3;
    } 

// Calculate spread rate from Cheney et al. (1998) (converting spread rate to m/s from km/hr) 
if ( wind_speed >= 5.0 )
    head_speed = (CF_Backing_Fast + CF_Wind_Fast * pow( (wind_speed - 5), 0.844 ) ) * moisture_coeff * curing_coeff / 3.6;
else
    head_speed = (CF_Backing_Slow + CF_Wind_Slow * wind_speed)* moisture_coeff * curing_coeff / 3.6; 

// Adjust speed based on canopy layer for Northern Australia grassland types (woodland and open forest) 
// Based on Cheney and Sullivan (2008)
head_speed = head_speed * speed_factor;

// Adjust for calculated speed coefficient for fire flanks
speed = head_speed * speed_fraction;
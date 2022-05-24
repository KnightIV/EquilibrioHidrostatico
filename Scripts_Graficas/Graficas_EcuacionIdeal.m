z = 0:1000:6e6      # altitude; m
p_z0 = 680          # pressure photosphere base; kg * m^-1 * s^-2
k_B = 1.380649e-23  # Boltzmann const.; m^2 * kg * s^-2 * K^-1
T_cor = 1.2e6       # coronal temperature; K
T_phot = 6_000      # photosphere temperature; K
z_t = 2e6           # photosphere-corona transition altitude; m
z_w = 0.2e6         # transition region width; m
g = 274             # Sun surface gravity acceleration; m * s^-2
m_p = 1.672621e-27  # proton mass; kg
dtc = T_phot / T_cor

################ Temperature ################
#---------- Function ----------#
tanh_exprsn_in = (z - z_t) / z_w
tanh_exprsn = (1 - dtc) * tanh(tanh_exprsn_in)
T_z = (1 / 2) * T_cor * (1 + dtc + tanh_exprsn)

#---------- Integral ----------#
## Numerator
num_fac1 = (dtc - 1) * z_w
num_ln_exp_pow1 = 2 * z_t / z_w
num_ln_exp_pow2 = 2 * z / z_w
num_ln_exp = e .^ (num_ln_exp_pow1 .- num_ln_exp_pow2)
num_ln = log((dtc * num_ln_exp) + 1)
num = num_fac1 * num_ln

## Denominator
den = 2 * T_cor * dtc

temperature_integral = (num / den) .+ (z / T_cor)
#temperature_integral = temperature_integral .- temperature_integral(1)
################################################

################ Pressure ################
p_fac1_eq_state = - ((m_p * g) / (2 * k_B))
p_z = p_z0 * exp(p_fac1_eq_state * temperature_integral)
################################################

################ Density ################
density_num = m_p * p_z
density_den = 2 * k_B * T_z
density_z = density_num ./ density_den
#########################################

################ Graph code ################
subplot(3, 1, 1)
plot_data_helper(z, p_z, ...
          {"Altitude (z)", "m"}, {"Pressure", "kg * m^{-1} * s^{-2}"}, ...
          "Pressure as function of altitude", ...
          "b", true)
                    
subplot(3, 1, 2)          
plot_data_helper(z, T_z, ...
          {"Altitude (z)", "m"}, {"Temperature", "K"}, ...
          "Temperature as function of altitude", ...
          "r", false)
          
subplot(3, 1, 3)
plot_data_helper(z, density_z, ...
          {"Altitude (z)", "m"}, {"Density", "kg * m^{-3}"}, ...
          "Density as function of altitude", ...
          "g", true) 
&RunSpecification
 strict_nl_parsing=.true.
 verbosity="moderate"
/
&GlobalResource
 dictionary="{{ resources }}/dictionary_cosmo.txt"
 grib_definition_path="{{ resources }}/eccodes_definitions_cosmo",
                        "{{ resources }}/eccodes_definitions_vendor"
 grib2_sample="{{ resources }}/eccodes_samples/COSMO_GRIB2_default.tmpl"
 rttov_coefs_path="{{ resources }}/rttov_coefficients"
/
&GlobalSettings
 default_model_name="cosmo-1e"
/
&ModelSpecification
 model_name="cosmo-1e"
 earth_axis_large=6371229.
 earth_axis_small=6371229.
/

&Process
 in_file  = "{{ file.inputi }}"
 out_file = "{{ file.output }}",
 tstart = 0, tstop = 0, tincr = 1
 out_type = "NETCDF"
/
&Process in_field = "U_10M", poper='n2geog' /
&Process in_field = "V_10M", poper='n2geog' /

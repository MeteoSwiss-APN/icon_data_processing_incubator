# History

## [0.2.0-rc3] (2023-12-18)

- Added vertical interpolation operator `interpolate_k2any`
- Testing data marker determines the test data that should be used
- Removed the `system_definitions` module


## [0.2.0-rc2] (2023-12-11)

- Added `product.run_products` function
- Updated field mappings
- Data cache will inject the step and number in the request depending on the file pattern
- `mars.Request` now supports multiple params


## [0.2.0-rc1] (2023-12-01)

- Added support for the FDB data source
- Added support for running tests on balfrin (FDB only)
- Added modules
    - `config`: manage configuration
    - `data_cache`: support fieldextra testing from FDB
    - `data_source`: enable reading from files or from FDB
    - `mars`: mars request validation
    - `product`: product description
    - `tasking`: dask delayed wrapper
- Added operators
    - atmo
    - gis
    - lateral_operators
    - regrid
    - relhum
    - time_operators
    - wind
- Changed: flexpart operator no longer uses cosmo naming conventions
- **Breaking:** removed the `load_fields` method from the GribReader class
- **Breaking:** the `load` method of GribReader class now takes a mapping of labels to requests and returns a mapping of labels to `xarray.DataArray`


## [0.1.0] (2023-07-11)

- Added operators
    * brn
    * curl
    * destagger
    * diff
    * flexpart (for IFS model output)
    * hzerocl
    * omega_slope
    * pot_vortic
    * rho
    * theta
    * thetav
    * time_rate
    * interpolate_k2p
    * interpolate_k2theta
    * minmax_k
    * integrate_k
- Added ninjo_k2th product
- Added GRIB data loader based on earthkit-data

[0.2.0-rc3]: https://github.com/MeteoSwiss-APN/icon_data_processing_incubator/compare/v0.2.0-rc2..v0.2.0-rc3
[0.2.0-rc2]: https://github.com/MeteoSwiss-APN/icon_data_processing_incubator/compare/v0.2.0-rc1..v0.2.0-rc2
[0.2.0-rc1]: https://github.com/MeteoSwiss-APN/icon_data_processing_incubator/compare/v0.1.0..v0.2.0-rc1
[0.1.0]: https://github.com/MeteoSwiss-APN/icon_data_processing_incubator/tree/v0.1.0

def mask_under_pval(field, ps):
    for p_val in field.coords["z"].data:
        field.loc[dict(z=p_val)] = field.sel(dict(z=p_val)).where(ps >= p_val * 100)
    return field

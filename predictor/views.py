from django.shortcuts import render

from . import ml


def index(request):
    context = {"locations": ml.LOCATION_CHOICES}

    if request.method == "POST":
        location = (request.POST.get("location") or "").strip()
        area = (request.POST.get("area") or "").strip()
        bed = (request.POST.get("bed") or "").strip()
        bath = (request.POST.get("bath") or "").strip()

        # keep what the user typed so the form stays filled after submit
        context["submitted"] = {
            "location": location, "area": area, "bed": bed, "bath": bath,
        }

        errors = []
        if not location:
            errors.append("Choose a location.")
        try:
            area_val = float(area)
            if area_val <= 0:
                errors.append("Area must be greater than 0.")
        except ValueError:
            area_val = None
            errors.append("Area must be a number, e.g. 1200.")
        try:
            bed_val = int(bed)
            if bed_val < 0:
                errors.append("Bedrooms can't be negative.")
        except ValueError:
            bed_val = None
            errors.append("Bedrooms must be a whole number.")
        try:
            bath_val = int(bath)
            if bath_val < 0:
                errors.append("Bathrooms can't be negative.")
        except ValueError:
            bath_val = None
            errors.append("Bathrooms must be a whole number.")

        if errors:
            context["errors"] = errors
        else:
            rent = ml.predict_rent(location, area_val, bed_val, bath_val)
            context["result"] = {
                "rent": round(rent),
                "known": ml.is_known_location(location),
                "area": area_val,
                "bed": bed_val,
                "bath": bath_val,
                "location": location,
            }

    return render(request, "predictor/index.html", context)

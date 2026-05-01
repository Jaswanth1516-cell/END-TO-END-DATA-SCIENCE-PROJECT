from pydantic import BaseModel, Field


class IrisFeatures(BaseModel):
    sepal_length_cm: float = Field(..., alias="sepal length (cm)")
    sepal_width_cm: float = Field(..., alias="sepal width (cm)")
    petal_length_cm: float = Field(..., alias="petal length (cm)")
    petal_width_cm: float = Field(..., alias="petal width (cm)")

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "sepal length (cm)": 5.1,
                "sepal width (cm)": 3.5,
                "petal length (cm)": 1.4,
                "petal width (cm)": 0.2,
            }
        }


class PredictionResponse(BaseModel):
    predicted_class: int
    predicted_label: str
    probabilities: dict[str, float]

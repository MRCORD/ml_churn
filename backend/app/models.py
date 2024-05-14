from pydantic import BaseModel, Field, conint, confloat

class ChurnDataRequest(BaseModel):
    customerID: str = Field('6464-UIAEA', min_length=1, strip_whitespace=True)
    gender: str = Field(..., pattern="^(Female|Male)$")
    SeniorCitizen: conint(ge=0, le=1)
    Partner: str = Field(..., pattern="^(No|Yes)$")
    Dependents: str = Field(..., pattern="^(No|Yes)$")
    tenure: conint(ge=0)  # Assuming max_tenure is validated elsewhere
    PhoneService: str = Field(..., pattern="^(No|Yes)$")
    MultipleLines: str = Field(..., pattern="^(No|Yes)$")
    InternetService: str = Field(..., pattern="^(No|DSL|Fiber optic)$")
    OnlineSecurity: str = Field(..., pattern="^(No|Yes)$")
    OnlineBackup: str = Field(..., pattern="^(No|Yes)$")
    DeviceProtection: str = Field(..., pattern="^(No|Yes)$")
    TechSupport: str = Field(..., pattern="^(No|Yes)$")
    StreamingTV: str = Field(..., pattern="^(No|Yes)$")
    StreamingMovies: str = Field(..., pattern="^(No|Yes)$")
    Contract: str = Field(..., pattern="^(Month-to-month|One year|Two year)$")
    PaperlessBilling: str = Field(..., pattern="^(No|Yes)$")
    PaymentMethod: str = Field(..., pattern="^(Electronic check|Mailed check|Bank transfer \\(automatic\\)|Credit card \\(automatic\\))$")
    MonthlyCharges: confloat(ge=0.0)  # Assuming max_monthly_charges is validated elsewhere
    TotalCharges: confloat(ge=0.0)  # Assuming max_total_charges is validated elsewhere
    
class ShapRequest(BaseModel):
    customer_id: str = Field('0988-JRWWP', min_length=1, strip_whitespace=True)
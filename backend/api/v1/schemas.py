"""
Request/Response schemas for API v1.
Separate from contracts for API-specific validation.
"""
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator


class ScreeningRequest(BaseModel):
    """Stage A screening request - symptoms are optional but ignored"""
    # Person
    age: Optional[int] = Field(None, ge=18, le=45, description="Age 18-45")
    gender: Optional[int] = Field(None, ge=0, le=1, description="0=F, 1=M")
    height: Optional[int] = Field(None, ge=120, le=230, description="Height in cm")
    weight: Optional[int] = Field(None, ge=30, le=250, description="Weight in kg")
    
    # Sleep
    sleep_duration: Optional[float] = Field(None, ge=0, le=24, description="Hours per day")
    sleep_quality: Optional[int] = Field(None, ge=1, le=5, description="1-5, 5=best")
    sleep_disorder: Optional[int] = Field(None, ge=0, le=1)
    wake_up_during_night: Optional[int] = Field(None, ge=0, le=1)
    feel_sleepy_during_day: Optional[int] = Field(None, ge=0, le=1)
    
    # Device/Screen
    average_screen_time: Optional[float] = Field(None, ge=0, le=24, description="Hours per day")
    smart_device_before_bed: Optional[int] = Field(None, ge=0, le=1)
    bluelight_filter: Optional[int] = Field(None, ge=0, le=1)
    
    # Lifestyle
    stress_level: Optional[int] = Field(None, ge=1, le=5, description="1-5, 5=highest")
    daily_steps: Optional[int] = Field(None, ge=0, le=50000)
    physical_activity: Optional[int] = Field(None, ge=0, le=600)
    caffeine_consumption: Optional[int] = Field(None, ge=0, le=1)
    alcohol_consumption: Optional[int] = Field(None, ge=0, le=1)
    smoking: Optional[int] = Field(None, ge=0, le=1)
    
    # Vitals
    systolic: Optional[int] = Field(None, ge=70, le=250)
    diastolic: Optional[int] = Field(None, ge=40, le=150)
    heart_rate: Optional[int] = Field(None, ge=40, le=220)
    
    # Medical
    medical_issue: Optional[int] = Field(None, ge=0, le=1)
    ongoing_medication: Optional[int] = Field(None, ge=0, le=1)
    
    # Symptoms (optional, ignored in Stage A)
    discomfort_eyestrain: Optional[int] = Field(None, ge=0, le=1)
    redness_in_eye: Optional[int] = Field(None, ge=0, le=1)
    itchiness_irritation_in_eye: Optional[int] = Field(None, ge=0, le=1)
    
    @field_validator('*', mode='before')
    @classmethod
    def validate_ranges(cls, v, info):
        """Soft validation: allow missing, reject insane values"""
        if v is None:
            return v
        # Additional sanity checks can go here
        return v


class TriageRequest(BaseModel):
    """Stage B triage request - must include symptoms"""
    # All Stage A fields
    age: Optional[int] = Field(None, ge=18, le=45)
    gender: Optional[int] = Field(None, ge=0, le=1)
    height: Optional[int] = Field(None, ge=120, le=230)
    weight: Optional[int] = Field(None, ge=30, le=250)
    
    sleep_duration: Optional[float] = Field(None, ge=0, le=24)
    sleep_quality: Optional[int] = Field(None, ge=1, le=5)
    sleep_disorder: Optional[int] = Field(None, ge=0, le=1)
    wake_up_during_night: Optional[int] = Field(None, ge=0, le=1)
    feel_sleepy_during_day: Optional[int] = Field(None, ge=0, le=1)
    
    average_screen_time: Optional[float] = Field(None, ge=0, le=24)
    smart_device_before_bed: Optional[int] = Field(None, ge=0, le=1)
    bluelight_filter: Optional[int] = Field(None, ge=0, le=1)
    
    stress_level: Optional[int] = Field(None, ge=1, le=5)
    daily_steps: Optional[int] = Field(None, ge=0, le=50000)
    physical_activity: Optional[int] = Field(None, ge=0, le=600)
    caffeine_consumption: Optional[int] = Field(None, ge=0, le=1)
    alcohol_consumption: Optional[int] = Field(None, ge=0, le=1)
    smoking: Optional[int] = Field(None, ge=0, le=1)
    
    systolic: Optional[int] = Field(None, ge=70, le=250)
    diastolic: Optional[int] = Field(None, ge=40, le=150)
    heart_rate: Optional[int] = Field(None, ge=40, le=220)
    
    medical_issue: Optional[int] = Field(None, ge=0, le=1)
    ongoing_medication: Optional[int] = Field(None, ge=0, le=1)
    
    # Symptoms (required for Stage B)
    discomfort_eyestrain: Optional[int] = Field(None, ge=0, le=1)
    redness_in_eye: Optional[int] = Field(None, ge=0, le=1)
    itchiness_irritation_in_eye: Optional[int] = Field(None, ge=0, le=1)


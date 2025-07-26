import math
from typing import Dict, Any, Union

class CalculatorTool:
    """Calculator tool for medical calculations like BMI and dosage."""
    
    def calculate(self, calculation_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a medical calculation.
        
        Args:
            calculation_type: Type of calculation to perform
            parameters: Parameters for the calculation
            
        Returns:
            Result of the calculation
        """
        try:
            if calculation_type == "bmi":
                return self.calculate_bmi(
                    parameters.get("weight_kg"),
                    parameters.get("height_m")
                )
            elif calculation_type == "dosage":
                return self.calculate_dosage(
                    parameters.get("weight_kg"),
                    parameters.get("dosage_per_kg")
                )
            elif calculation_type == "body_surface_area":
                return self.calculate_bsa(
                    parameters.get("weight_kg"),
                    parameters.get("height_cm")
                )
            else:
                return {"error": f"Unknown calculation type: {calculation_type}"}
                
        except Exception as e:
            return {"error": f"Error performing calculation: {str(e)}"}
    
    def calculate_bmi(self, weight_kg: float, height_m: float) -> Dict[str, Any]:
        """
        Calculate Body Mass Index (BMI).
        
        Args:
            weight_kg: Weight in kilograms
            height_m: Height in meters
            
        Returns:
            BMI calculation result
        """
        if weight_kg is None or height_m is None:
            return {"error": "Both weight_kg and height_m are required"}
        
        if weight_kg <= 0 or height_m <= 0:
            return {"error": "Weight and height must be positive values"}
        
        bmi = weight_kg / (height_m ** 2)
        
        # BMI categories
        if bmi < 18.5:
            category = "Underweight"
        elif 18.5 <= bmi < 25:
            category = "Normal weight"
        elif 25 <= bmi < 30:
            category = "Overweight"
        else:
            category = "Obesity"
        
        return {
            "bmi": round(bmi, 2),
            "category": category,
            "interpretation": f"A BMI of {round(bmi, 2)} falls into the {category} category."
        }
    
    def calculate_dosage(self, weight_kg: float, dosage_per_kg: float) -> Dict[str, Any]:
        """
        Calculate medication dosage based on weight.
        
        Args:
            weight_kg: Patient weight in kilograms
            dosage_per_kg: Dosage per kilogram
            
        Returns:
            Dosage calculation result
        """
        if weight_kg is None or dosage_per_kg is None:
            return {"error": "Both weight_kg and dosage_per_kg are required"}
        
        if weight_kg <= 0 or dosage_per_kg <= 0:
            return {"error": "Weight and dosage must be positive values"}
        
        total_dosage = weight_kg * dosage_per_kg
        
        return {
            "total_dosage": round(total_dosage, 2),
            "weight_kg": weight_kg,
            "dosage_per_kg": dosage_per_kg,
            "unit": "mg"  # Default unit
        }
    
    def calculate_bsa(self, weight_kg: float, height_cm: float) -> Dict[str, Any]:
        """
        Calculate Body Surface Area (BSA) using the DuBois formula.
        
        Args:
            weight_kg: Weight in kilograms
            height_cm: Height in centimeters
            
        Returns:
            BSA calculation result
        """
        if weight_kg is None or height_cm is None:
            return {"error": "Both weight_kg and height_cm are required"}
        
        if weight_kg <= 0 or height_cm <= 0:
            return {"error": "Weight and height must be positive values"}
        
        # DuBois formula: BSA = 0.007184 × Weight^0.425 × Height^0.725
        bsa = 0.007184 * (weight_kg ** 0.425) * (height_cm ** 0.725)
        
        return {
            "bsa": round(bsa, 2),
            "unit": "m²",
            "formula": "DuBois formula"
        }
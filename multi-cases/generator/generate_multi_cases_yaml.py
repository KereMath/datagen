# -*- coding: utf-8 -*-
"""
Multi-Case YAML Configuration Generator for Time Series
Generates comprehensive test cases combining base models with characteristics
"""

import os
import itertools
import yaml
from pathlib import Path

# TSGen'deki base model üretim fonksiyonlarına göre düzeltilmiş mapping
BASE_MODEL_MAPPING = {
    # Stationary base seriler (generate_stationary_base_series)
    "AR": {"type": "Stationary", "base_function": "generate_stationary_base_series", "distribution": "ar"},
    "MA": {"type": "Stationary", "base_function": "generate_stationary_base_series", "distribution": "ma"},
    "ARMA": {"type": "Stationary", "base_function": "generate_stationary_base_series", "distribution": "arma"},
    "White Noise": {"type": "Stationary", "base_function": "generate_stationary_base_series", "distribution": "white_noise"},
    
    # Stochastic trend seriler (generate_stochastic_trend)
    "RW": {"type": "Stochastic", "base_function": "generate_stochastic_trend", "distribution": "rw"},
    "RW with Drift": {"type": "Stochastic", "base_function": "generate_stochastic_trend", "distribution": "rwd"},
    "ARI": {"type": "Stochastic", "base_function": "generate_stochastic_trend", "distribution": "ari"},
    "IMA": {"type": "Stochastic", "base_function": "generate_stochastic_trend", "distribution": "ima"},
    "ARIMA": {"type": "Stochastic", "base_function": "generate_stochastic_trend", "distribution": "arima"},
    
    # Seasonal base seriler (generate_seasonality_from_base_series)
    "SARMA": {"type": "Seasonal", "base_function": "generate_seasonality_from_base_series", "distribution": "sarma"},
    "SARIMA": {"type": "Seasonal", "base_function": "generate_seasonality_from_base_series", "distribution": "sarima"},
    
    # Volatility seriler (generate_volatility)
    "ARCH": {"type": "Volatility", "base_function": "generate_volatility", "distribution": "arch"},
    "GARCH": {"type": "Volatility", "base_function": "generate_volatility", "distribution": "garch"},
}

CHARACTERISTICS_MAPPING = {
    # Trend Characteristics
    "Linear Trend (Up)": {
        "function": "generate_deterministic_trend_linear",
        "params": {"sign": 1, "scale_factor": 1},
        "output_flag": "is_linear_trend_active",
        "plot_type": "trend",
        "variation_name": "linear_trend_up"
    },
    "Linear Trend (Down)": {
        "function": "generate_deterministic_trend_linear", 
        "params": {"sign": -1, "scale_factor": 1},
        "output_flag": "is_linear_trend_active",
        "plot_type": "trend",
        "variation_name": "linear_trend_down"
    },
    "Quadratic Trend (Up)": {
        "function": "generate_deterministic_trend_quadratic",
        "params": {"sign": 1, "scale_factor": 1},
        "output_flag": "is_quadratic_trend_active", 
        "plot_type": "trend",
        "variation_name": "quadratic_trend_up"
    },
    "Quadratic Trend (Down)": {
        "function": "generate_deterministic_trend_quadratic",
        "params": {"sign": -1, "scale_factor": 1},
        "output_flag": "is_quadratic_trend_active",
        "plot_type": "trend", 
        "variation_name": "quadratic_trend_down"
    },
    "Cubic Trend (Up)": {
        "function": "generate_deterministic_trend_cubic",
        "params": {"sign": 1, "scale_factor": 1},
        "output_flag": "is_cubic_trend_active",
        "plot_type": "trend",
        "variation_name": "cubic_trend_up"
    },
    "Cubic Trend (Down)": {
        "function": "generate_deterministic_trend_cubic",
        "params": {"sign": -1, "scale_factor": 1},
        "output_flag": "is_cubic_trend_active", 
        "plot_type": "trend",
        "variation_name": "cubic_trend_down"
    },
    "Exponential Trend (Up)": {
        "function": "generate_deterministic_trend_exponential",
        "params": {"sign": 1, "scale_factor": 1},
        "output_flag": "is_exponential_trend_active",
        "plot_type": "trend",
        "variation_name": "exponential_trend_up"
    },
    "Exponential Trend (Down)": {
        "function": "generate_deterministic_trend_exponential",
        "params": {"sign": -1, "scale_factor": 1},
        "output_flag": "is_exponential_trend_active",
        "plot_type": "trend",
        "variation_name": "exponential_trend_down"
    },
    "Damped Trend (Up)": {
        "function": "generate_deterministic_trend_damped",
        "params": {"sign": 1, "scale_factor": 1},
        "output_flag": "is_damped_trend_active",
        "plot_type": "trend",
        "variation_name": "damped_trend_up"
    },
    "Damped Trend (Down)": {
        "function": "generate_deterministic_trend_damped",
        "params": {"sign": -1, "scale_factor": 1},
        "output_flag": "is_damped_trend_active",
        "plot_type": "trend",
        "variation_name": "damped_trend_down"
    },
    
    # Seasonality Characteristics
    "Single Seasonality": {
        "function": "generate_single_seasonality",
        "params": {"scale_factor": 3},
        "output_flag": "is_seasonality_active",
        "plot_type": "seasonality",
        "variation_name": "single_seasonality"
    },
    "Multiple Seasonality": {
        "function": "generate_multiple_seasonality", 
        "params": {"num_components": 2, "scale_factor": 3},
        "output_flag": "is_multiple_seasonality_active",
        "plot_type": "seasonality",
        "variation_name": "multiple_seasonality"
    },
    
    # Anomaly Characteristics
    "Point Anomaly (Beginning)": {
        "function": "generate_point_anomaly",
        "params": {"location": "beginning", "scale_factor": 1, "return_debug": True},
        "output_flag": "is_point_anomaly_active",
        "plot_type": "anomaly",
        "variation_name": "point_1events_beginning"
    },
    "Point Anomaly (Middle)": {
        "function": "generate_point_anomaly",
        "params": {"location": "middle", "scale_factor": 1, "return_debug": True},
        "output_flag": "is_point_anomaly_active",
        "plot_type": "anomaly",
        "variation_name": "point_1events_middle"
    },
    "Point Anomaly (End)": {
        "function": "generate_point_anomaly",
        "params": {"location": "end", "scale_factor": 1, "return_debug": True},
        "output_flag": "is_point_anomaly_active",
        "plot_type": "anomaly",
        "variation_name": "point_1events_end"
    },
    "Multiple Point Anomalies": {
        "function": "generate_point_anomalies",
        "params": {"scale_factor": 1, "return_debug": True},
        "output_flag": "is_point_anomaly_active",
        "plot_type": "anomaly",
        "variation_name": "point_multiple_random_locations"
    },
    "Collective Anomalies (Beginning)": {
        "function": "generate_collective_anomalies",
        "params": {"num_anomalies": 1, "location": "beginning", "scale_factor": 1, "return_debug": True},
        "output_flag": "is_collective_anomaly_active",
        "plot_type": "anomaly",
        "variation_name": "collective_1events_beginning"
    },
    "Collective Anomalies (Middle)": {
        "function": "generate_collective_anomalies", 
        "params": {"num_anomalies": 1, "location": "middle", "scale_factor": 1, "return_debug": True},
        "output_flag": "is_collective_anomaly_active",
        "plot_type": "anomaly",
        "variation_name": "collective_1events_middle"
    },
    "Collective Anomalies (End)": {
        "function": "generate_collective_anomalies",
        "params": {"num_anomalies": 1, "location": "end", "scale_factor": 1, "return_debug": True},
        "output_flag": "is_collective_anomaly_active",
        "plot_type": "anomaly",
        "variation_name": "collective_1events_end"
    },
    "Multiple Collective Anomalies": {
        "function": "generate_collective_anomalies",
        "params": {"scale_factor": 1, "return_debug": True},
        "output_flag": "is_collective_anomaly_active",
        "plot_type": "anomaly", 
        "variation_name": "collective_multiple_random_locations"
    },
    "Contextual Anomalies": {
        "function": "generate_contextual_anomalies",
        "params": {"scale_factor": 1, "return_debug": True},
        "output_flag": "is_contextual_anomaly_active",
        "plot_type": "anomaly",
        "variation_name": "contextual_1events_random_locations"
    },
    
    # Structural Break Characteristics
    "Mean Shift (Beginning)": {
        "function": "generate_mean_shift",
        "params": {"num_breaks": 1, "location": "beginning", "scale_factor": 1, "return_debug": True},
        "output_flag": "is_mean_shift_active",
        "plot_type": "break",
        "variation_name": "mean_shift_1events_beginning"
    },
    "Mean Shift (Middle)": {
        "function": "generate_mean_shift",
        "params": {"num_breaks": 1, "location": "middle", "scale_factor": 1, "return_debug": True},
        "output_flag": "is_mean_shift_active",
        "plot_type": "break",
        "variation_name": "mean_shift_1events_middle"
    },
    "Mean Shift (End)": {
        "function": "generate_mean_shift",
        "params": {"num_breaks": 1, "location": "end", "scale_factor": 1, "return_debug": True},
        "output_flag": "is_mean_shift_active",
        "plot_type": "break",
        "variation_name": "mean_shift_1events_end"
    },
    "Multiple Mean Shifts": {
        "function": "generate_mean_shift",
        "params": {"scale_factor": 1, "return_debug": True},
        "output_flag": "is_mean_shift_active",
        "plot_type": "break",
        "variation_name": "mean_shift_multiple_random_locations"
    },
    "Variance Shift (Beginning)": {
        "function": "generate_variance_shift",
        "params": {"num_breaks": 1, "location": "beginning", "scale_factor": 1, "return_debug": True},
        "output_flag": "is_variance_shift_active",
        "plot_type": "break",
        "variation_name": "variance_shift_1events_beginning"
    },
    "Variance Shift (Middle)": {
        "function": "generate_variance_shift",
        "params": {"num_breaks": 1, "location": "middle", "scale_factor": 1, "return_debug": True},
        "output_flag": "is_variance_shift_active",
        "plot_type": "break",
        "variation_name": "variance_shift_1events_middle"
    },
    "Variance Shift (End)": {
        "function": "generate_variance_shift",
        "params": {"num_breaks": 1, "location": "end", "scale_factor": 1, "return_debug": True},
        "output_flag": "is_variance_shift_active",
        "plot_type": "break",
        "variation_name": "variance_shift_1events_end"
    },
    "Multiple Variance Shifts": {
        "function": "generate_variance_shift",
        "params": {"scale_factor": 1, "return_debug": True},
        "output_flag": "is_variance_shift_active",
        "plot_type": "break",
        "variation_name": "variance_shift_multiple_random_locations"
    },
    "Trend Shift - Direction Change (Beginning)": {
        "function": "generate_trend_shift",
        "params": {"signs": [1], "num_breaks": 1, "location": "beginning", "change_type": "direction_change", "scale_factor": 1, "return_debug": True},
        "output_flag": "is_trend_shift_active",
        "plot_type": "break",
        "variation_name": "trend_shift_direction_1events_beginning"
    },
    "Trend Shift - Direction Change (Middle)": {
        "function": "generate_trend_shift",
        "params": {"signs": [1], "num_breaks": 1, "location": "middle", "change_type": "direction_change", "scale_factor": 1, "return_debug": True},
        "output_flag": "is_trend_shift_active",
        "plot_type": "break",
        "variation_name": "trend_shift_direction_1events_middle"
    },
    "Trend Shift - Direction Change (End)": {
        "function": "generate_trend_shift",
        "params": {"signs": [1], "num_breaks": 1, "location": "end", "change_type": "direction_change", "scale_factor": 1, "return_debug": True},
        "output_flag": "is_trend_shift_active",
        "plot_type": "break",
        "variation_name": "trend_shift_direction_1events_end"
    },
    "Trend Shift - Magnitude Change (Beginning)": {
        "function": "generate_trend_shift",
        "params": {"signs": [-1], "num_breaks": 1, "location": "beginning", "change_type": "magnitude_change", "scale_factor": 1, "return_debug": True},
        "output_flag": "is_trend_shift_active",
        "plot_type": "break",
        "variation_name": "trend_shift_magnitude_1events_beginning"
    },
    "Trend Shift - Magnitude Change (Middle)": {
        "function": "generate_trend_shift",
        "params": {"signs": [-1], "num_breaks": 1, "location": "middle", "change_type": "magnitude_change", "scale_factor": 1, "return_debug": True},
        "output_flag": "is_trend_shift_active",
        "plot_type": "break",
        "variation_name": "trend_shift_magnitude_1events_middle"
    },
    "Trend Shift - Magnitude Change (End)": {
        "function": "generate_trend_shift",
        "params": {"signs": [-1], "num_breaks": 1, "location": "end", "change_type": "magnitude_change", "scale_factor": 1, "return_debug": True},
        "output_flag": "is_trend_shift_active",
        "plot_type": "break",
        "variation_name": "trend_shift_magnitude_1events_end"
    },
    "Trend Shift - Direction & Magnitude Change (Beginning)": {
        "function": "generate_trend_shift",
        "params": {"signs": [1], "num_breaks": 1, "location": "beginning", "change_type": "direction_and_magnitude_change", "scale_factor": 1, "return_debug": True},
        "output_flag": "is_trend_shift_active",
        "plot_type": "break",
        "variation_name": "trend_shift_dir_mag_1events_beginning"
    },
    "Trend Shift - Direction & Magnitude Change (Middle)": {
        "function": "generate_trend_shift",
        "params": {"signs": [1], "num_breaks": 1, "location": "middle", "change_type": "direction_and_magnitude_change", "scale_factor": 1, "return_debug": True},
        "output_flag": "is_trend_shift_active",
        "plot_type": "break",
        "variation_name": "trend_shift_dir_mag_1events_middle"
    },
    "Trend Shift - Direction & Magnitude Change (End)": {
        "function": "generate_trend_shift",
        "params": {"signs": [1], "num_breaks": 1, "location": "end", "change_type": "direction_and_magnitude_change", "scale_factor": 1, "return_debug": True},
        "output_flag": "is_trend_shift_active",
        "plot_type": "break",
        "variation_name": "trend_shift_dir_mag_1events_end"
    }
}

def create_folder_name(base_name, char_details):
    """Create a clean folder name from base model and characteristic details."""
    folder_parts = [base_name.lower().replace(" ", "_")]
    
    for char_detail in char_details:
        if 'variation_name' in char_detail:
            folder_parts.append(char_detail['variation_name'])
    
    return "_".join(folder_parts)

def create_case_name(base_name, char_names):
    """Create a readable case name."""
    char_part = "_".join([name.replace(" ", "_") for name in char_names])
    return f"{base_name}_{char_part}"

def is_valid_combination(base_info, characteristics):
    """Check if the combination of base model and characteristics is valid."""
    char_names = [char["name"] for char in characteristics]
    
    for char_name in char_names:
        # Seasonal karakteristikleri stationary base'e uygulayamayız
        if (char_name in ["Single Seasonality", "Multiple Seasonality"] and 
            base_info["type"] != "Seasonal"):
            return False
            
        # Contextual anomalileri sadece seasonal serilere uygulayabiliriz  
        if (char_name == "Contextual Anomalies" and 
            base_info["type"] != "Seasonal"):
            return False
    
    # Aynı tip karakteristikleri birlikte kullanmayalım (trend + trend, anomaly + anomaly vs.)
    char_types = []
    for char_name in char_names:
        if char_name in CHARACTERISTICS_MAPPING:
            char_types.append(CHARACTERISTICS_MAPPING[char_name]["plot_type"])
    
    if len(set(char_types)) != len(char_types):  # Duplicate types found
        return False
    
    return True

def generate_multi_cases_yaml():
    """Generate comprehensive multi-cases YAML configuration."""
    
    # Output directory setup
    script_dir = Path(__file__).parent
    output_file = script_dir / "multi_cases.yaml"
    
    cases = []
    
    # Single characteristics - her base model ile her karakteristik
    print("Generating single characteristic cases...")
    for base_name, base_info in BASE_MODEL_MAPPING.items():
        for char_name, char_detail in CHARACTERISTICS_MAPPING.items():
            
            # Validity check
            if not is_valid_combination(base_info, [{"name": char_name}]):
                continue
                
            case_name = create_case_name(base_name, [char_name])
            folder_name = create_folder_name(base_name, [char_detail])
            
            case = {
                "name": case_name,
                "base_model": {
                    "function": base_info["base_function"],
                    "params": {"distribution": base_info["distribution"]}
                },
                "base_model_info": base_info,
                "characteristics": [char_detail],
                "output_folder": folder_name,
                "num_series_per_case": 5,
                "workflow": "sequential",
                "base_plot_title": f"{base_name} + {char_name.lower()}"
            }
            cases.append(case)
    
    # Multi-characteristics - limited combinations to prevent explosion
    print("Generating multi-characteristic cases...")
    
    # Prioritized combinations  
    prioritized_combinations = [
        # Trend + Anomaly combinations
        (["Linear Trend (Up)", "Linear Trend (Down)"], 
         ["Point Anomaly (Middle)", "Multiple Point Anomalies", "Collective Anomalies (Middle)"]),
        
        # Trend + Break combinations  
        (["Quadratic Trend (Up)", "Quadratic Trend (Down)"],
         ["Mean Shift (Middle)", "Variance Shift (Middle)", "Trend Shift - Direction Change (Middle)"]),
        
        # Seasonality + Anomaly combinations (only for Seasonal base models)
        (["Single Seasonality"], 
         ["Contextual Anomalies", "Point Anomaly (Middle)", "Collective Anomalies (Middle)"]),
         
        # Seasonality + Break combinations (only for Seasonal base models)
        (["Multiple Seasonality"],
         ["Mean Shift (Middle)", "Variance Shift (Middle)"]),
    ]
    
    for base_name, base_info in BASE_MODEL_MAPPING.items():
        for trend_chars, other_chars in prioritized_combinations:
            
            # Skip seasonality combinations for non-seasonal bases
            if any("Seasonality" in char for char in trend_chars) and base_info["type"] != "Seasonal":
                continue
            if any("Contextual" in char for char in other_chars) and base_info["type"] != "Seasonal":
                continue
                
            for trend_char in trend_chars[:2]:  # Limit to 2 trend variations
                for other_char in other_chars[:3]:  # Limit to 3 other variations
                    
                    char_names = [trend_char, other_char]
                    
                    # Validity check
                    char_objects = [{"name": name} for name in char_names]
                    if not is_valid_combination(base_info, char_objects):
                        continue
                    
                    char_details = [CHARACTERISTICS_MAPPING[name] for name in char_names]
                    case_name = create_case_name(base_name, char_names)
                    folder_name = create_folder_name(base_name, char_details)
                    
                    case = {
                        "name": case_name,
                        "base_model": {
                            "function": base_info["base_function"],
                            "params": {"distribution": base_info["distribution"]}
                        },
                        "base_model_info": base_info,
                        "characteristics": char_details,
                        "output_folder": folder_name,
                        "num_series_per_case": 3,  # Fewer for multi-char cases
                        "workflow": "sequential", 
                        "base_plot_title": f"{base_name} + {' + '.join([name.lower() for name in char_names])}"
                    }
                    cases.append(case)
    
    # Write YAML file
    with open(output_file, 'w') as f:
        yaml.dump(cases, f, default_flow_style=False, sort_keys=False, indent=2)
    
    print(f"Generated {len(cases)} cases")
    print(f"Configuration saved to: {output_file}")
    
    # Summary statistics
    single_char_cases = sum(1 for case in cases if len(case["characteristics"]) == 1)
    multi_char_cases = sum(1 for case in cases if len(case["characteristics"]) > 1)
    
    print(f"\nSummary:")
    print(f"- Single characteristic cases: {single_char_cases}")
    print(f"- Multi characteristic cases: {multi_char_cases}")
    print(f"- Total cases: {len(cases)}")
    
    # Base model distribution
    base_counts = {}
    for case in cases:
        base_name = case["name"].split("_")[0]
        base_counts[base_name] = base_counts.get(base_name, 0) + 1
    
    print(f"\nPer base model:")
    for base_name, count in sorted(base_counts.items()):
        print(f"- {base_name}: {count} cases")

if __name__ == "__main__":
    generate_multi_cases_yaml()
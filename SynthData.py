import numpy as np
import pandas as pd

# Set random seed for reproducibility (global)
np.random.seed(42)

# Number of production orders to generate
num_orders = 10000  # Large enough for ML testing

# Define product types with fixed num_operations required
products = {
    'A': 5,
    'B': 6,
    'C': 4,
    'D': 7,
    'E': 5,
    'F': 8,
    'G': 3,
    'H': 6,
    'I': 4,
    'J': 5
}

product_types = list(products.keys())

# Define 10 production lines with distinct characteristics, including max_operations
# Each line has base parameters: 
# - max_operations: integer (maximum number of operations the line can handle)
# - base_operators: integer (number of operators)
# - base_shifts: integer (number of shifts per day)
# - base_wip: integer (units in work-in-process, e.g., pieces)
# - base_throughput: float (units per day, e.g., pieces per day)
lines = {
    1: {'max_operations': 6, 'base_operators': 4, 'base_shifts': 2, 'base_wip': 50, 'base_throughput': 100},
    2: {'max_operations': 7, 'base_operators': 5, 'base_shifts': 2, 'base_wip': 60, 'base_throughput': 110},
    3: {'max_operations': 5, 'base_operators': 3, 'base_shifts': 1, 'base_wip': 40, 'base_throughput': 90},
    4: {'max_operations': 8, 'base_operators': 6, 'base_shifts': 3, 'base_wip': 70, 'base_throughput': 120},
    5: {'max_operations': 6, 'base_operators': 4, 'base_shifts': 2, 'base_wip': 55, 'base_throughput': 95},
    6: {'max_operations': 9, 'base_operators': 7, 'base_shifts': 3, 'base_wip': 80, 'base_throughput': 130},
    7: {'max_operations': 4, 'base_operators': 3, 'base_shifts': 1, 'base_wip': 35, 'base_throughput': 85},
    8: {'max_operations': 7, 'base_operators': 5, 'base_shifts': 2, 'base_wip': 65, 'base_throughput': 115},
    9: {'max_operations': 5, 'base_operators': 4, 'base_shifts': 2, 'base_wip': 45, 'base_throughput': 100},
    10: {'max_operations': 6, 'base_operators': 5, 'base_shifts': 2, 'base_wip': 55, 'base_throughput': 105}
}

# Intervention (DBR from TOC): Applied to lines 1 and 2
intervention_lines = [1, 2]

# Pre-compute consistent performance and efficiency adjustment factors for each line-product pair
# This creates a "binômio" (line-product) effect: some lines perform better/worse on specific products
performance_factors = {}      # Multiplier for throughput (e.g., linha mais "velha" ou menos compatível tem fator menor)
efficiency_adjusts = {}       # Small multiplier for efficiency (variação adicional por produto)

for line in range(1, 11):
    for prod in product_types:
        # Unique seed per line-product pair for consistent random values across runs
        seed = line * 100 + (ord(prod) - ord('A'))
        rng = np.random.default_rng(seed)
        
        # Performance factor for throughput: mean 1.0, std 0.15 → variation ±15-30%
        # Clip to min 0.6 to avoid unrealistically low performance
        perf_factor = rng.normal(1.0, 0.15)
        performance_factors[(line, prod)] = max(0.6, perf_factor)
        
        # Efficiency adjustment: smaller variation, mean 1.0, std 0.05 → ±5-10%
        eff_adjust = rng.normal(1.0, 0.05)
        efficiency_adjusts[(line, prod)] = eff_adjust

# Track last product per line for setup time calculation
last_product_per_line = {}

# Generate dataset
data = []

for order_id in range(1, num_orders + 1):
    # Randomly assign a line to each order (balanced distribution)
    line_id = np.random.choice(range(1, 11))
    line_params = lines[line_id]
    
    # Treatment: 1 if line has intervention, 0 otherwise
    treatment = 1 if line_id in intervention_lines else 0
    
    # Order size: lognormal distribution around 100-500 units (e.g., pieces)
    order_size = int(np.random.lognormal(mean=5, sigma=0.5))  # Around 100-500
    
    # Select product_type that fits the line's max_operations
    compatible_products = [pt for pt in product_types if products[pt] <= line_params['max_operations']]
    if not compatible_products:
        product_type = 'G'  # Fallback to low-ops product
    else:
        product_type = np.random.choice(compatible_products)
    
    # Num_operations: fixed per product (number of operations used, unitless)
    num_operations = products[product_type]
    
    # Operators: base + some variation (number of people)
    operators = max(1, int(np.random.normal(line_params['base_operators'], 1)))
    
    # Shifts: base + variation (1-3) (number of shifts per day)
    shifts = max(1, min(3, int(np.random.normal(line_params['base_shifts'], 0.5))))
    
    # Efficiency: base by treatment (dimensionless, 0-1)
    if treatment:
        efficiency = np.clip(np.random.normal(0.85, 0.05), 0.8, 1.0)
    else:
        efficiency = np.clip(np.random.normal(0.5, 0.1), 0.01, 0.6)
    
    # Apply line-product specific adjustment to efficiency (small variation)
    eff_adjust = efficiency_adjusts[(line_id, product_type)]
    efficiency *= eff_adjust
    # Re-clip to maintain treatment effect
    if treatment:
        efficiency = np.clip(efficiency, 0.8, 1.0)
    else:
        efficiency = np.clip(efficiency, 0.01, 0.6)
    
    # Throughput: base with small variance (clipped to positive), multiplied by efficiency, shifts, and line-product performance factor
    throughput_base = line_params['base_throughput']
    base_variation = np.clip(np.random.normal(throughput_base, 10), 1, None)
    perf_factor = performance_factors[(line_id, product_type)]
    throughput = base_variation * efficiency * shifts * perf_factor
    
    # WIP: base + noise, lower for treated (units in process, e.g., pieces)
    wip_base = line_params['base_wip']
    if treatment:
        wip = int(np.random.normal(wip_base * 0.8, 5))  # 20% lower on average
    else:
        wip = int(np.random.normal(wip_base, 10))
    
    # Setup time: if product different from previous on this line, random positive (in days); else 0
    if line_id in last_product_per_line and product_type != last_product_per_line[line_id]:
        setup_time = np.clip(np.random.normal(0.1, 0.02), 0.01, None)  # e.g., ~0.1 days (about 2.4 hours)
    else:
        setup_time = 0.0
    # Update last product for this line
    last_product_per_line[line_id] = product_type
    
    # Leadtime (in days): Based on Little's Law: LT = WIP / Throughput (days)
    leadtime = wip / throughput
    
    # Other variables: e.g., defect rate (lower with treatment) (dimensionless, 0-1 or fraction defective)
    defect_rate = np.clip(np.random.normal(0.05, 0.02), 0.0, 0.2)
    if treatment:
        defect_rate *= 0.7  # Reduced defects
    
    # Append row
    data.append({
        'order_id': order_id,
        'order_size': order_size,
        'product_type': product_type,
        'line_id': line_id,
        'treatment': treatment,
        'num_operations': num_operations,
        'operators': operators,
        'shifts': shifts,
        'efficiency': efficiency,
        'wip': wip,
        'setup_time': setup_time,
        'leadtime': leadtime,
        'throughput': throughput,
        'defect_rate': defect_rate
    })

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('synthetic_production_dataset.csv', index=False)

print("=" * 80)
print("DATASET SINTÉTICO GERADO COM SUCESSO")
print("=" * 80)
print(f"\nTotal de ordens: {len(df)}")
print(f"Arquivo salvo: synthetic_production_dataset.csv")
print("\nPrimeiras 10 linhas:")
print(df.head(10).to_string())
print("\nEstatísticas descritivas:")
print(df.describe().to_string())

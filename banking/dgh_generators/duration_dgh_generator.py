import pandas as pd

duration_dgh_rows_granular = []

for value in range(0, 4920):
    # Level 1: 50s intervals
    if value < 50:
        l1 = "[0,50["
    elif value < 100:
        l1 = "[50,100["
    elif value < 150:
        l1 = "[100,150["
    elif value < 200:
        l1 = "[150,200["
    elif value < 250:
        l1 = "[200,250["
    elif value < 300:
        l1 = "[250,300["
    elif value < 350:
        l1 = "[300,350["
    elif value < 400:
        l1 = "[350,400["
    elif value < 450:
        l1 = "[400,450["
    elif value < 500:
        l1 = "[450,500["
    elif value < 550:
        l1 = "[500,550["
    elif value < 600:
        l1 = "[550,600["
    elif value < 650:
        l1 = "[600,650["
    elif value < 700:
        l1 = "[650,700["
    elif value < 750:
        l1 = "[700,750["
    elif value < 800:
        l1 = "[750,800["
    elif value < 850:
        l1 = "[800,850["
    elif value < 900:
        l1 = "[850,900["
    elif value < 950:
        l1 = "[900,950["
    elif value < 1000:
        l1 = "[950,1000["
    elif value < 1050:
        l1 = "[1000,1050["
    elif value < 1100:
        l1 = "[1050,1100["
    elif value < 1150:
        l1 = "[1100,1150["
    elif value < 1200:
        l1 = "[1150,1200["
    elif value < 1500:
        l1 = "[1200,1500["
    elif value < 1800:
        l1 = "[1500,1800["
    elif value < 2100:
        l1 = "[1800,2100["
    elif value < 2400:
        l1 = "[2100,2400["
    elif value < 2700:
        l1 = "[2400,2700["
    elif value < 3000:
        l1 = "[2700,3000["
    else:
        l1 = ">=3000"

    # Level 2:
    if value < 100:
        l2 = "[0,100["
    elif value < 200:
        l2 = "[100,200["
    elif value < 300:
        l2 = "[200,300["
    elif value < 400:
        l2 = "[300,400["
    elif value < 500:
        l2 = "[400,500["
    elif value < 600:
        l2 = "[500,600["
    elif value < 700:
        l2 = "[600,700["
    elif value < 800:
        l2 = "[700,800["
    elif value < 900:
        l2 = "[800,900["
    elif value < 1000:
        l2 = "[900,1000["
    elif value < 1100:
        l2 = "[1000,1100["
    elif value < 1200:
        l2 = "[1100,1200["
    elif value < 1300:
        l2 = "[1200,1300["
    elif value < 1400:
        l2 = "[1300,1400["
    elif value < 1500:
        l2 = "[1400,1500["
    elif value < 1600:
        l2 = "[1500,1600["
    elif value < 1700:
        l2 = "[1600,1700["
    elif value < 1800:
        l2 = "[1700,1800["
    elif value < 1900:
        l2 = "[1800,1900["
    elif value < 2000:
        l2 = "[1900,2000["
    elif value < 2100:
        l2 = "[2000,2100["
    elif value < 2200:
        l2 = "[2100,2200["
    elif value < 2300:
        l2 = "[2200,2300["
    elif value < 2400:
        l2 = "[2300,2400["
    elif value < 2500:
        l2 = "[2400,2500["
    elif value < 2600:
        l2 = "[2500,2600["
    elif value < 2700:
        l2 = "[2600,2700["
    elif value < 2800:
        l2 = "[2700,2800["
    elif value < 2900:
        l2 = "[2800,2900["
    elif value < 3000:
        l2 = "[2900,3000["
    else:
        l2 = ">=3000"

    # Level 3: bucket by 5 groups
    if value < 250:
        l3 = "[0,250["
    elif value < 500:
        l3 = "[250,500["
    elif value < 750:
        l3 = "[500,750["
    elif value < 1000:
        l3 = "[750,1000["
    elif value < 1250:
        l3 = "[1000,1250["
    elif value < 1500:
        l3 = "[1250,1500["
    elif value < 1750:
        l3 = "[1500,1750["
    elif value < 2000:
        l3 = "[1750,2000["
    elif value < 2250:
        l3 = "[2000,2250["
    elif value < 2500:
        l3 = "[2250,2500["
    elif value < 2750:
        l3 = "[2500,2750["
    elif value < 3000:
        l3 = "[2750,3000["
    else:
        l3 = ">=3000"

    # Level 4: quartile buckets
    if value < 500:
        l4 = "[0,500["
    elif value < 1000:
        l4 = "[500,1000["
    elif value < 1500:
        l4 = "[1000,1500["
    elif value < 2000:
        l4 = "[1500,2000["
    elif value < 2500:
        l4 = "[2000,2500["
    elif value < 3000:
        l4 = "[2500,3000["
    else:
        l4 = ">=3000"

    # Level 4 and 5
    if value < 3000:
        l5 = "[0,3000["
    else:
        l5 = ">=3000"
    l6 = "*"

    duration_dgh_rows_granular.append([value, l1, l2, l3, l4, l5, l5, l6])

# Create and export granular DGH
duration_dgh_granular = pd.DataFrame(duration_dgh_rows_granular)
duration_dgh_granular.to_csv("hierarchies/duration.csv", index=False)

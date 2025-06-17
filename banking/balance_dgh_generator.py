import pandas as pd

balance_dgh_rows = []

for value in range(-8019, 102150):
    # Level 1: fine-grained (500 interval)
    if value < -8000:
        l1 = "<-8000"
    else:
        bucket_start = (value // 500) * 500
        l1 = f"[{bucket_start},{bucket_start + 500}["

    # Level 2: 1000 interval
    if value < -8000:
        l2 = "<-8000"
    else:
        bucket_start = (value // 1000) * 1000
        l2 = f"[{bucket_start},{bucket_start + 1000}["

    # Level 3: 2000 interval
    if value < -8000:
        l3 = "<-8000"
    else:
        bucket_start = (value // 2000) * 2000
        l3 = f"[{bucket_start},{bucket_start + 2000}["

    # Level 4: 4000 interval
    if value < -8000:
        l4 = "<-8000"
    else:
        bucket_start = (value // 4000) * 4000
        l4 = f"[{bucket_start},{bucket_start + 4000}["

    # Level 5: 10000 interval
    if value < -10000:
        l5 = "<-10000"
    else:
        bucket_start = (value // 10000) * 10000
        l5 = f"[{bucket_start},{bucket_start + 10000}["

    # Level 6: general bucket
    if value < 0:
        l6 = "<0"
    elif value < 10000:
        l6 = "[0,10000["
    elif value < 50000:
        l6 = "[10000,50000["
    else:
        l6 = ">=50000"

    # Level 7: root
    l7 = "*"

    balance_dgh_rows.append([value, l1, l2, l3, l4, l5, l6, l7])

# Create and export the DGH
balance_dgh = pd.DataFrame(balance_dgh_rows)
balance_dgh.to_csv("hierarchies/balance.csv", index=False)

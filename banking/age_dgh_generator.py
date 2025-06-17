import pandas as pd

age_dgh_rows = []
for value in range(17, 97):
	# Level 1:
	if value < 20:
		l1 = "[15,20["
	elif value < 25:
		l1 = "[20,25["
	elif value < 30:
		l1 = "[25,30["
	elif value < 35:
		l1 = "[30,35["
	elif value < 40:
		l1 = "[35,40["
	elif value < 45:
		l1 = "[40,45["
	elif value < 50:
		l1 = "[45,50["
	elif value < 55:
		l1 = "[50,55["
	elif value < 60:
		l1 = "[55,60["
	elif value < 65:
		l1 = "[60,65["
	elif value < 70:
		l1 = "[65,70["
	elif value < 75:
		l1 = "[70,75["
	elif value < 80:
		l1 = "[75,80["
	else:
		l1 = ">=80"

	# Level 2:
	if value < 20:
		l2 = "[10,20["
	elif value < 30:
		l2 = "[20,30["
	elif value < 40:
		l2 = "[30,40["
	elif value < 50:
		l2 = "[40,50["
	elif value < 60:
		l2 = "[50,60["
	elif value < 70:
		l2 = "[60,70["
	elif value < 80:
		l2 = "[70,80["
	else:
		l2 = ">=80"

	# Level 3:
	if value < 20:
		l3 = "[0,20["
	elif value < 40:
		l3 = "[20,40["
	elif value < 60:
		l3 = "[40,60["
	elif value < 80:
		l3 = "[60,80["
	else:
		l3 = ">=80"

	# Level 4 and 5:
	if value < 40:
		l4 = "[0,40["
		l5 = "[0,80["
	elif value < 80:
		l4 = "[40,80["
		l5 = "[0,80["
	else:
		l4 = ">=80"
		l5 = ">=80"
	l6 = "*"

	age_dgh_rows.append([value, l1, l2, l3, l4, l5, l6])

# Create and export granular DGH
age_dgh = pd.DataFrame(age_dgh_rows)
age_dgh.to_csv("hierarchies/age.csv", index=False)



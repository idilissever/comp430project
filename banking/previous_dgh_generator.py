import pandas as pd

previous_dgh_rows = []

for value in range(-1, 276):
	if value < 0:
		l1 = "-1"
	elif value < 10:
		l1 = "[0,10["
	elif value < 20:
		l1 = "[10,20["
	elif value < 30:
		l1 = "[20,30["
	elif value < 40:
		l1 = "[30,40["
	elif value < 50:
		l1 = "[40,50["
	elif value < 60:
		l1 = "[50,60["
	elif value < 70:
		l1 = "[60,70["
	elif value < 80:
		l1 = "[70,80["
	elif value < 90:
		l1 = "[80,90["
	elif value < 100:
		l1 = "[90,100["
	elif value < 110:
		l1 = "[100,110["
	elif value < 120:
		l1 = "[110,120["
	elif value < 130:
		l1 = "[120,130["
	elif value < 140:
		l1 = "[130,140["
	elif value < 150:
		l1 = "[140,150["
	elif value < 160:
		l1 = "[150,160["
	elif value < 170:
		l1 = "[160,170["
	elif value < 180:
		l1 = "[170,180["
	elif value < 190:
		l1 = "[180,190["
	elif value < 200:
		l1 = "[190,200["
	else:
		l1 = ">=200"

	if value < 0:
		l2 = "-1"
	elif value < 20:
		l2 = "[0,20["
	elif value < 40:
		l2 = "[20,40["
	elif value < 60:
		l2 = "[40,60["
	elif value < 80:
		l2 = "[60,80["
	elif value < 100:
		l2 = "[80,100["
	elif value < 120:
		l2 = "[100,120["
	elif value < 140:
		l2 = "[120,140["
	elif value < 160:
		l2 = "[140,160["
	elif value < 180:
		l2 = "[160,180["
	elif value < 200:
		l2 = "[180,200["
	else:
		l2 = ">=200"

	if value < 0:
		l3 = "-1"
	elif value < 40:
		l3 = "[0,40["
	elif value < 80:
		l3 = "[40,80["
	elif value < 120:
		l3 = "[80,120["
	elif value < 160:
		l3 = "[120,160["
	elif value < 200:
		l3 = "[160,200["
	else:
		l3 = ">=200"

	if value < 0:
		l4 = "-1"
	elif value < 80:
		l4 = "[0,80["
	elif value < 160:
		l4 = "[80,160["
	elif value < 200:
		l4 = "[160,200["
	else:
		l4 = ">=200"

	if value < 0:
		l5 = "-1"
	else:
		l5 = ">=0"

	l6 = "*"

	previous_dgh_rows.append([value, l1, l2, l3, l4, l5, l6])

previous_dgh = pd.DataFrame(previous_dgh_rows)
previous_dgh.to_csv("hierarchies/previous.csv", index=False)
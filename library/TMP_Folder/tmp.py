# File used for quick test of snippet of codes

div = 2.81
div_gr = 3.71
annual_inv = 100
years = 10

tot = annual_inv
tot_div = 0
div_per_year = [div]
inv_per_year = [annual_inv]
for i in range(years):
    tmp_div = 0
    for j in range(len(div_per_year)):
        tmp_div += round(inv_per_year[j] * div_per_year[j] / 100, 2)
        div_per_year[j] *= (1 + div_gr / 100)
    tot_div += tmp_div
    print(i, j)
    print("Year: ", i + 1)
    print("\tTot invested: ", tot)
    print("\tDiv annuale : ", tmp_div)
    print("\tTot dividend: ", tot_div)
    tot += annual_inv + tmp_div
    inv_per_year.append(annual_inv + tmp_div)
    div_per_year.append(div)
# print(inv_per_year)
# print(div_per_year)


L = [x for x in range(2)]
L.extend([5,5,5,5,6,7,8,9])
def b_s_d(L,tar):
    left = 0
    right = len(L) - 1
    while tar != L[left] or L[right] != tar:
        if tar == L[left]:
            right -=1
        elif tar == L[right]:
            left +=1
        else:
            left +=1
            right -=1
    return [left, right]
#L = [4,5,6]
print(L)
print(b_s_d(L,5))
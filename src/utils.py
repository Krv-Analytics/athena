

def convert_to_float(value):
    try:
        return float(value.replace(',', ''))
    except:
        return value
    
def abs_amount(id):
    if id <0:
        return 0
    else:
        return id
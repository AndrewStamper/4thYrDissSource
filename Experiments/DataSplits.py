def training_data_list():
    training_scan_numbers = []

    for number in range(1, 40):
        for letter in ["L", "R"]:
            if number < 10:
                t_number = "0" + str(number)
            else:
                t_number = str(number)
            training_scan_numbers.append("A0" + t_number + letter)
    training_scan_numbers.remove("A003L")
    training_scan_numbers.remove("A005R")
    training_scan_numbers.remove("A018R")

    return training_scan_numbers


def validation_data_list():
    validation_scan_numbers = []
    for number in range(40, 50):
        for letter in ["L", "R"]:
            if number < 10:
                t_number = "0" + str(number)
            else:
                t_number = str(number)
            validation_scan_numbers.append("A0" + t_number + letter)

    validation_scan_numbers.remove("A041R")
    return validation_scan_numbers

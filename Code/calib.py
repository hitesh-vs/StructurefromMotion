with open(r"C:\Users\farha\OneDrive\Desktop\P2Data\P2Data\matching1.txt", "r") as file:
    for line in file:  # Iterates through each line in the file
        print(line.strip())  # Removes any leading/trailing whitespace, including newlines

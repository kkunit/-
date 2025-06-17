import management_system as ms

def get_student_details_from_input():
    name = input("Enter student name: ")
    student_id = input("Enter student ID: ")
    while True:
        try:
            age = int(input("Enter student age: "))
            break
        except ValueError:
            print("Invalid age. Please enter a number.")
    gender = input("Enter student gender: ")
    contact_information = input("Enter contact information: ")
    return name, student_id, age, gender, contact_information

DEFAULT_DATA_FILE = "students.csv"

def main():
    # Attempt to load data on startup
    print(f"Attempting to load data from {DEFAULT_DATA_FILE}...")
    if ms.import_from_csv(DEFAULT_DATA_FILE):
        print(f"Data loaded successfully from {DEFAULT_DATA_FILE}.")
    else:
        # This message will also cover FileNotFoundError handled by import_from_csv
        print(f"Could not load data from {DEFAULT_DATA_FILE}. Starting with an empty dataset or no pre-existing file.")

    while True:
        print("\nStudent Management System")
        print("1. Add Student")
        print("2. Delete Student")
        print("3. Modify Student")
        print("4. Search Student")
        print("5. Display All Students")
        print("6. Import from CSV")
        print("7. Export to CSV")
        print("8. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            print("\n--- Add New Student ---")
            name, student_id, age, gender, contact = get_student_details_from_input()
            student = ms.add_student(name, student_id, age, gender, contact)
            if student:
                print(f"Student '{student.name}' added successfully with ID: {student.student_id}")
            else:
                # This case should ideally not happen with current add_student logic
                print("Failed to add student.")

        elif choice == '2':
            print("\n--- Delete Student ---")
            student_id = input("Enter student ID to delete: ")
            if ms.delete_student(student_id):
                print(f"Student with ID '{student_id}' deleted successfully.")
            else:
                print(f"Student with ID '{student_id}' not found.")

        elif choice == '3':
            print("\n--- Modify Student ---")
            student_id = input("Enter student ID to modify: ")
            # Check if student exists
            existing_student = ms.search_student(student_id=student_id)
            if not existing_student or not isinstance(existing_student, ms.Student): # type check for single student
                print(f"Student with ID '{student_id}' not found.")
                continue

            print(f"Modifying student: {existing_student.name} (ID: {existing_student.student_id})")
            name = input(f"Enter new name (current: {existing_student.name}, press Enter to keep): ") or None

            age_str = input(f"Enter new age (current: {existing_student.age}, press Enter to keep): ")
            age = None
            if age_str:
                while True:
                    try:
                        age = int(age_str)
                        break
                    except ValueError:
                        age_str = input("Invalid age. Please enter a number (or press Enter to keep current): ")
                        if not age_str:
                            break # Keep current if empty input after error

            gender = input(f"Enter new gender (current: {existing_student.gender}, press Enter to keep): ") or None
            contact = input(f"Enter new contact information (current: {existing_student.contact_information}, press Enter to keep): ") or None

            modified_student = ms.modify_student(student_id, name, age, gender, contact)
            if modified_student:
                print(f"Student '{modified_student.name}' updated successfully.")
            else:
                # This case should not be reached if student was found initially
                print(f"Failed to modify student with ID '{student_id}'.")


        elif choice == '4':
            print("\n--- Search Student ---")
            search_type = input("Search by (1) ID or (2) Name? Enter 1 or 2: ")
            if search_type == '1':
                student_id = input("Enter student ID: ")
                student = ms.search_student(student_id=student_id)
                if student and isinstance(student, ms.Student):
                    print(f"Found student: Name: {student.name}, ID: {student.student_id}, Age: {student.age}, Gender: {student.gender}, Contact: {student.contact_information}")
                else:
                    print(f"Student with ID '{student_id}' not found.")
            elif search_type == '2':
                name = input("Enter student name (or part of name): ")
                results = ms.search_student(name=name)
                if results and isinstance(results, list):
                    if not results:
                        print(f"No students found with name matching '{name}'.")
                    else:
                        print(f"Found {len(results)} student(s):")
                        for student in results:
                            print(f"- Name: {student.name}, ID: {student.student_id}, Age: {student.age}, Gender: {student.gender}, Contact: {student.contact_information}")
                else: # Should be an empty list if no results
                     print(f"No students found with name matching '{name}'.")
            else:
                print("Invalid search type.")

        elif choice == '5':
            print("\n--- Display All Students ---")
            ms.display_all_students()

        elif choice == '6':
            print("\n--- Import from CSV ---")
            filename = input("Enter CSV filename to import from (e.g., students.csv): ")
            if ms.import_from_csv(filename):
                print(f"Data imported from '{filename}' successfully (check console for any row-specific errors).")
            else:
                print(f"Failed to import data from '{filename}'. Check if file exists and is readable.")

        elif choice == '7':
            print("\n--- Export to CSV ---")
            filename = input("Enter CSV filename to export to (e.g., students_export.csv): ")
            if ms.export_to_csv(filename):
                print(f"Data exported to '{filename}' successfully.")
            else:
                print(f"Failed to export data to '{filename}'.")

        elif choice == '8':
            print(f"Saving data to {DEFAULT_DATA_FILE}...")
            if ms.export_to_csv(DEFAULT_DATA_FILE):
                print(f"Data saved successfully to {DEFAULT_DATA_FILE}.")
            else:
                print(f"Failed to save data to {DEFAULT_DATA_FILE}.")
            print("Exiting Student Management System. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()

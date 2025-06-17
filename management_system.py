import csv
from student import Student

students: list[Student] = []

def add_student(name: str, student_id: str, age: int, gender: str, contact_information: str) -> Student:
    """
    Creates a new Student object and adds it to the students list.

    Args:
        name: The name of the student.
        student_id: The ID of the student.
        age: The age of the student.
        gender: The gender of the student.
        contact_information: The contact information of the student.

    Returns:
        The newly created Student object.
    """
    student = Student(name, student_id, age, gender, contact_information)
    students.append(student)
    return student

def delete_student(student_id: str) -> bool:
    """
    Deletes a student from the students list based on their student_id.

    Args:
        student_id: The ID of the student to delete.

    Returns:
        True if a student was deleted, False otherwise.
    """
    global students
    initial_length = len(students)
    students = [student for student in students if student.student_id != student_id]
    return len(students) != initial_length

def modify_student(student_id: str, name: str = None, age: int = None, gender: str = None, contact_information: str = None) -> Student | None:
    """
    Modifies an existing student's information.

    Args:
        student_id: The ID of the student to modify.
        name: The new name of the student (optional).
        age: The new age of the student (optional).
        gender: The new gender of the student (optional).
        contact_information: The new contact information of the student (optional).

    Returns:
        The modified Student object if found and updated, None otherwise.
    """
    for student in students:
        if student.student_id == student_id:
            if name is not None:
                student.name = name
            if age is not None:
                student.age = age
            if gender is not None:
                student.gender = gender
            if contact_information is not None:
                student.contact_information = contact_information
            return student
    return None

def search_student(student_id: str = None, name: str = None) -> Student | list[Student] | None:
    """
    Searches for students by student_id or name.

    Args:
        student_id: The ID of the student to search for (optional).
        name: The name of the student to search for (case-insensitive, optional).

    Returns:
        A single Student object if student_id is provided and found.
        A list of Student objects if name is provided and matches are found.
        None if student_id is provided but not found, or if neither student_id nor name is provided.
        An empty list if name is provided but no matches are found.
    """
    if student_id is not None:
        for student in students:
            if student.student_id == student_id:
                return student
        return None
    elif name is not None:
        matched_students = [student for student in students if name.lower() in student.name.lower()]
        return matched_students
    return None

def display_all_students():
    """
    Displays all student records in a readable format.
    Prints "No students in the system." if the list is empty.
    """
    if not students:
        print("No students in the system.")
        return

    for student in students:
        print(f"Name: {student.name}, ID: {student.student_id}, Age: {student.age}, Gender: {student.gender}, Contact: {student.contact_information}")

def export_to_csv(filename: str) -> bool:
    """
    Exports all student records to a CSV file.

    Args:
        filename: The name of the CSV file to create.

    Returns:
        True if export was successful, False otherwise.
    """
    try:
        with open(filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            # Write header row
            csv_writer.writerow(['Name', 'Student ID', 'Age', 'Gender', 'Contact Information'])
            # Write student data
            for student in students:
                csv_writer.writerow([student.name, student.student_id, student.age, student.gender, student.contact_information])
        return True
    except IOError:
        print(f"Error: Could not write to file {filename}")
        return False

def import_from_csv(filename: str) -> bool:
    """
    Imports student records from a CSV file.

    Args:
        filename: The name of the CSV file to import.

    Returns:
        True if import process was attempted (even with row errors), False if a major error like file not found occurs.
    """
    try:
        with open(filename, 'r', newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            header = next(csv_reader) # Skip header row

            for row_num, row in enumerate(csv_reader, 1):
                if len(row) != 5:
                    print(f"Error in row {row_num}: Expected 5 columns, got {len(row)}. Skipping row: {row}")
                    continue

                name, student_id, age_str, gender, contact_information = row

                try:
                    age = int(age_str)
                except ValueError:
                    print(f"Error in row {row_num}: Could not convert age '{age_str}' to integer. Skipping row: {row}")
                    continue

                # Use the existing add_student function to ensure consistency
                add_student(name, student_id, age, gender, contact_information)
        return True
    except FileNotFoundError:
        print(f"Error: File not found {filename}")
        return False
    except csv.Error as e:
        print(f"Error reading CSV file {filename}: {e}")
        return False
    except IOError:
        print(f"Error: Could not read file {filename}")
        return False

import unittest
import os
import csv
from io import StringIO
from contextlib import redirect_stdout

# Important: Assuming management_system.py and student.py are in the same directory or accessible via PYTHONPATH
import management_system as ms
from student import Student

class TestManagementSystem(unittest.TestCase):

    def setUp(self):
        """Clear the students list before each test."""
        ms.students.clear()
        self.test_csv_file = "test_students.csv"
        self.test_export_file = "test_export_students.csv"

    def tearDown(self):
        """Clean up any created files."""
        for f in [self.test_csv_file, self.test_export_file]:
            if os.path.exists(f):
                os.remove(f)

    def test_add_student(self):
        student = ms.add_student("Alice Smith", "S1001", 20, "Female", "alice@example.com")
        self.assertIn(student, ms.students)
        self.assertEqual(len(ms.students), 1)
        self.assertEqual(ms.students[0].name, "Alice Smith")

    def test_delete_student(self):
        s1 = ms.add_student("Bob Johnson", "S1002", 22, "Male", "bob@example.com")

        # Delete existing student
        self.assertTrue(ms.delete_student("S1002"))
        self.assertNotIn(s1, ms.students)
        self.assertEqual(len(ms.students), 0)

        # Try deleting non-existing student
        self.assertFalse(ms.delete_student("S9999"))
        self.assertEqual(len(ms.students), 0)

    def test_modify_student(self):
        ms.add_student("Charlie Brown", "S1003", 21, "Male", "charlie@example.com")

        modified_student = ms.modify_student("S1003", name="Charles Brown", age=22, contact_information="cb@newdomain.com")
        self.assertIsNotNone(modified_student)
        self.assertEqual(modified_student.name, "Charles Brown")
        self.assertEqual(modified_student.age, 22)
        self.assertEqual(modified_student.gender, "Male") # Unchanged
        self.assertEqual(modified_student.contact_information, "cb@newdomain.com")

        # Check if the list contains the updated student
        self.assertEqual(ms.students[0].name, "Charles Brown")

        # Try modifying non-existing student
        non_existent_modify = ms.modify_student("S9999", name="Ghost")
        self.assertIsNone(non_existent_modify)

    def test_search_student(self):
        s1 = ms.add_student("David Lee", "S1004", 23, "Male", "david@example.com")
        s2 = ms.add_student("Diana Ross", "S1005", 20, "Female", "diana@example.com")
        s3 = ms.add_student("David Bowie", "S1006", 25, "Male", "david.bowie@example.com")

        # Search by ID - found
        found_by_id = ms.search_student(student_id="S1004")
        self.assertEqual(found_by_id, s1)

        # Search by ID - not found
        not_found_by_id = ms.search_student(student_id="S9999")
        self.assertIsNone(not_found_by_id)

        # Search by Name - single exact match
        found_by_name_exact = ms.search_student(name="Diana Ross")
        self.assertIsInstance(found_by_name_exact, list)
        self.assertEqual(len(found_by_name_exact), 1)
        self.assertIn(s2, found_by_name_exact)

        # Search by Name - partial match, multiple results
        found_by_name_partial = ms.search_student(name="David")
        self.assertIsInstance(found_by_name_partial, list)
        self.assertEqual(len(found_by_name_partial), 2)
        self.assertIn(s1, found_by_name_partial)
        self.assertIn(s3, found_by_name_partial)

        # Search by Name - case-insensitive
        found_by_name_case = ms.search_student(name="diana ross")
        self.assertIsInstance(found_by_name_case, list)
        self.assertEqual(len(found_by_name_case), 1)
        self.assertIn(s2, found_by_name_case)

        # Search by Name - not found
        not_found_by_name = ms.search_student(name="NonExistentName")
        self.assertIsInstance(not_found_by_name, list)
        self.assertEqual(len(not_found_by_name), 0)

        # Search with no parameters
        self.assertIsNone(ms.search_student())


    def test_display_all_students(self):
        # Test with no students
        f = StringIO()
        with redirect_stdout(f):
            ms.display_all_students()
        output = f.getvalue().strip()
        self.assertEqual(output, "No students in the system.")

        # Test with a few students
        ms.add_student("Eve Adams", "S1007", 24, "Female", "eve@example.com")
        ms.add_student("Frank Sinatra", "S1008", 30, "Male", "frank@example.com")

        f = StringIO()
        with redirect_stdout(f):
            ms.display_all_students()
        output = f.getvalue().strip()

        self.assertIn("Name: Eve Adams, ID: S1007, Age: 24, Gender: Female, Contact: eve@example.com", output)
        self.assertIn("Name: Frank Sinatra, ID: S1008, Age: 30, Gender: Male, Contact: frank@example.com", output)

    def test_export_to_csv(self):
        ms.add_student("Grace Kelly", "S1009", 29, "Female", "grace@example.com")
        ms.add_student("Henry Ford", "S1010", 40, "Male", "henry@example.com")

        result = ms.export_to_csv(self.test_export_file)
        self.assertTrue(result)
        self.assertTrue(os.path.exists(self.test_export_file))

        with open(self.test_export_file, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
            self.assertEqual(header, ['Name', 'Student ID', 'Age', 'Gender', 'Contact Information'])

            data_row1 = next(reader)
            self.assertEqual(data_row1, ["Grace Kelly", "S1009", "29", "Female", "grace@example.com"])
            data_row2 = next(reader)
            self.assertEqual(data_row2, ["Henry Ford", "S1010", "40", "Male", "henry@example.com"])

        # Test export with no students (should create file with only header)
        ms.students.clear()
        result_empty = ms.export_to_csv(self.test_export_file) # Overwrite
        self.assertTrue(result_empty)
        with open(self.test_export_file, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
            self.assertEqual(header, ['Name', 'Student ID', 'Age', 'Gender', 'Contact Information'])
            with self.assertRaises(StopIteration): # No data rows
                next(reader)


    def test_import_from_csv(self):
        # Create a sample CSV file
        sample_data = [
            ['Name', 'Student ID', 'Age', 'Gender', 'Contact Information'],
            ['Ivy Green', 'S1011', '22', 'Female', 'ivy@example.com'],
            ['Jack Black', 'S1012', '35', 'Male', 'jack@example.com'],
            ['Bad AgeRow', 'S1013', 'Twenty', 'Male', 'bad@example.com'], # Malformed age
            ['Short Row', 'S1014', '28'] # Not enough columns
        ]
        with open(self.test_csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(sample_data)

        # Capture print output for error messages
        f_out = StringIO()
        with redirect_stdout(f_out):
            result = ms.import_from_csv(self.test_csv_file)

        output = f_out.getvalue()
        self.assertTrue(result)
        self.assertEqual(len(ms.students), 2) # Only two valid students should be added

        self.assertEqual(ms.students[0].name, "Ivy Green")
        self.assertEqual(ms.students[0].student_id, "S1011")
        self.assertEqual(ms.students[0].age, 22)

        self.assertEqual(ms.students[1].name, "Jack Black")
        self.assertEqual(ms.students[1].student_id, "S1012")
        self.assertEqual(ms.students[1].age, 35)

        # Check for error messages in output
        self.assertIn("Error in row 2: Could not convert age 'Twenty' to integer. Skipping row:", output) # Row 2 of data (after header)
        self.assertIn("Error in row 3: Expected 5 columns, got 3. Skipping row:", output) # Row 3 of data

        # Test import from non-existent file
        ms.students.clear() # Clear previous import
        f_out = StringIO()
        with redirect_stdout(f_out):
            result_not_found = ms.import_from_csv("non_existent_file.csv")
        output_not_found = f_out.getvalue()

        self.assertFalse(result_not_found)
        self.assertEqual(len(ms.students), 0)
        self.assertIn("Error: File not found non_existent_file.csv", output_not_found)

if __name__ == '__main__':
    unittest.main()

from flask import Flask, render_template, request, redirect, url_for
# import sqlite3 # No longer needed directly here
from database import get_db, init_db # Corrected import

app = Flask(__name__)
# DATABASE = 'students.db' # Defined in database.py

def query_db(query, args=(), one=False):
    conn = get_db()
    cur = conn.cursor() # Use cursor() method
    cur.execute(query, args)
    rv = cur.fetchall()
    conn.commit()
    # cur.close() # Cursor is closed when connection is closed if not explicitly closed before
    conn.close()
    return (rv[0] if rv else None) if one else rv

# --- CRUD Operations ---
def add_student_db(name, age, grade):
    query_db("INSERT INTO students (name, age, grade) VALUES (?, ?, ?)", (name, age, grade))

def get_all_students_db():
    return query_db("SELECT * FROM students")

def get_student_by_id_db(student_id):
    return query_db("SELECT * FROM students WHERE id = ?", (student_id,), one=True)

def update_student_db(student_id, name, age, grade):
    query_db("UPDATE students SET name = ?, age = ?, grade = ? WHERE id = ?", (name, age, grade, student_id))

def delete_student_db(student_id):
    query_db("DELETE FROM students WHERE id = ?", (student_id,))

# --- Flask Routes ---
@app.route('/')
def index():
    students = get_all_students_db()
    return render_template('index.html', students=students)

@app.route('/add', methods=['GET', 'POST'])
def add_student():
    if request.method == 'POST':
        name = request.form['name']
        age = request.form.get('age') # Use .get for optional fields
        grade = request.form.get('grade')

        # Basic validation for age if provided
        if age and not age.isdigit():
            # Handle error appropriately, e.g., flash a message or return error page
            return "Invalid age provided.", 400
        age = int(age) if age else None

        add_student_db(name, age, grade)
        return redirect(url_for('index'))
    return render_template('add_student.html')

@app.route('/edit/<int:student_id>', methods=['GET', 'POST'])
def edit_student(student_id):
    student = get_student_by_id_db(student_id)
    if not student:
        return "Student not found", 404

    if request.method == 'POST':
        name = request.form['name']
        age = request.form.get('age')
        grade = request.form.get('grade')

        if age and not age.isdigit():
            return "Invalid age provided.", 400
        age = int(age) if age else None

        update_student_db(student_id, name, age, grade)
        return redirect(url_for('index'))

    # Convert sqlite3.Row to a dict for easier template access if needed, or access by index/key
    # For simplicity, accessing by key directly in template should work with Row object.
    return render_template('edit_student.html', student=student)

@app.route('/delete/<int:student_id>', methods=['POST']) # Should be POST for destructive action
def delete_student(student_id):
    # Consider adding a check if student exists before deleting
    delete_student_db(student_id)
    return redirect(url_for('index'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)

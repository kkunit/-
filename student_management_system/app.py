from flask import Flask, render_template, request, redirect, url_for, flash
# import sqlite3 # No longer needed directly here
from database import get_db, init_db # Corrected import
import datetime # For footer date

app = Flask(__name__)
app.secret_key = 'your secret key' # Needed for flash messages
# DATABASE = 'students.db' # Defined in database.py

@app.context_processor
def inject_now():
    return {'now': datetime.datetime.utcnow()}

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

# --- CRUD Operations ---
def add_student_db(name, age, grade):
    query_db("INSERT INTO students (name, age, grade) VALUES (?, ?, ?)", (name, age, grade))

def get_all_students_db(search_name=None): # Added search_name parameter
    base_query = "SELECT * FROM students"
    args = []
    if search_name:
        base_query += " WHERE name LIKE ?"
        args.append(f"%{search_name}%")
    base_query += " ORDER BY id DESC" # Optional: order by ID or name
    return query_db(base_query, args)

def get_student_by_id_db(student_id):
    return query_db("SELECT * FROM students WHERE id = ?", (student_id,), one=True)

def update_student_db(student_id, name, age, grade): # Ensure this is here, was commented as "keep existing"
    query_db("UPDATE students SET name = ?, age = ?, grade = ? WHERE id = ?", (name, age, grade, student_id))

def delete_student_db(student_id): # Ensure this is here
    query_db("DELETE FROM students WHERE id = ?", (student_id,))

# --- Flask Routes ---
@app.route('/')
def index():
    search_name = request.args.get('search_name', '') # Ensure search_name is always a string
    page = request.args.get('page', 1, type=int)
    per_page = 5 # Students per page

    conn = get_db()
    cursor = conn.cursor()

    # Base query for counting
    count_query = "SELECT COUNT(id) FROM students"
    count_args = []
    if search_name:
        count_query += " WHERE name LIKE ?"
        count_args.append(f"%{search_name}%")

    total_students = cursor.execute(count_query, count_args).fetchone()[0]
    total_pages = (total_students + per_page - 1) // per_page # Ceiling division

    # Query for fetching students for the current page
    students_query = "SELECT * FROM students"
    students_args = []
    if search_name:
        students_query += " WHERE name LIKE ?"
        students_args.append(f"%{search_name}%")
    students_query += " ORDER BY id DESC LIMIT ? OFFSET ?"
    students_args.extend([per_page, (page - 1) * per_page])

    students_data = cursor.execute(students_query, students_args).fetchall()
    conn.close()

    # Ensure get_all_students_db is not used by index directly anymore, or update it to handle pagination
    # For now, index route handles its own pagination logic.
    # students = get_all_students_db(search_name=search_name) # This line is effectively replaced by the logic above

    return render_template('index.html', students=students_data, page=page, total_pages=total_pages, search_name=search_name)

@app.route('/add', methods=['GET', 'POST'])
def add_student():
    if request.method == 'POST':
        name = request.form['name']
        name = request.form['name'].strip() # Strip whitespace
        age_str = request.form.get('age', '').strip()
        grade = request.form.get('grade', '').strip()

        errors = {}
        if not name:
            errors['name'] = '姓名不能为空。'

        age = None
        if age_str:
            if not age_str.isdigit():
                errors['age'] = '年龄必须是有效的数字。'
            else:
                age = int(age_str)
                if not (0 <= age <= 150): # Reasonable age range
                    errors['age'] = '请输入一个合理的年龄。'

        if errors:
            for field, error_msg in errors.items():
                flash(error_msg, 'danger') # category 'danger' for Bootstrap alerts
            # Return the form with errors and original input
            return render_template('add_student.html', name=name, age=age_str, grade=grade, errors=errors), 400

        add_student_db(name, age, grade)
        flash('学生信息已成功添加！', 'success')
        return redirect(url_for('index'))
    return render_template('add_student.html', errors={}) # Pass empty errors dict for GET

@app.route('/edit/<int:student_id>', methods=['GET', 'POST'])
def edit_student(student_id):
    student = get_student_by_id_db(student_id)
    if not student:
        flash('未找到该学生。', 'warning')
        return redirect(url_for('index'))

    if request.method == 'POST':
        name = request.form['name'].strip()
        age_str = request.form.get('age', '').strip()
        grade = request.form.get('grade', '').strip()

        errors = {}
        if not name:
            errors['name'] = '姓名不能为空。'

        age = None
        if age_str:
            if not age_str.isdigit():
                errors['age'] = '年龄必须是有效的数字。'
            else:
                age = int(age_str)
                if not (0 <= age <= 150):
                    errors['age'] = '请输入一个合理的年龄。'

        if errors:
            for field, error_msg in errors.items():
                flash(error_msg, 'danger')
            # Pass current form values back to template
            current_values = {'id': student_id, 'name': name, 'age': age_str, 'grade': grade}
            return render_template('edit_student.html', student=current_values, errors=errors), 400

        update_student_db(student_id, name, age, grade)
        flash('学生信息已成功更新！', 'success')
        return redirect(url_for('index'))

    # For GET request, pass student data and empty errors
    return render_template('edit_student.html', student=student, errors={})

@app.route('/delete/<int:student_id>', methods=['POST']) # Should be POST for destructive action
def delete_student(student_id):
    student_to_delete = get_student_by_id_db(student_id)
    if student_to_delete:
        delete_student_db(student_id)
        flash(f"学生 '{student_to_delete['name']}' 已成功删除。", 'success')
    else:
        flash('尝试删除的学生不存在。', 'warning')
    return redirect(url_for('index'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)

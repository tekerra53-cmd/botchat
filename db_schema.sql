-- MySQL Schema for University Chatbot Knowledge Base
-- Run: mysql -u root -p university_kb < db_schema.sql

CREATE DATABASE IF NOT EXISTS university_kb;
USE university_kb;

-- Users table for admin auth
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role ENUM('admin') DEFAULT 'admin',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- FAQs table
CREATE TABLE faqs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    category VARCHAR(100) DEFAULT 'general',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Policies table
CREATE TABLE policies (
    id INT AUTO_INCREMENT PRIMARY KEY,
    title VARCHAR(200) NOT NULL,
    content TEXT NOT NULL,
    category VARCHAR(100) DEFAULT 'policy',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Academic Calendar
CREATE TABLE calendar (
    id INT AUTO_INCREMENT PRIMARY KEY,
    event_name VARCHAR(200) NOT NULL,
    event_date DATE NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Documents/Handbook for RAG (chunked content)
CREATE TABLE documents (
    id INT AUTO_INCREMENT PRIMARY KEY,
    title VARCHAR(200) NOT NULL,
    content TEXT NOT NULL,
    category VARCHAR(100) DEFAULT 'handbook',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Rich knowledge bank for common student questions
CREATE TABLE knowledge_base_entries (
    id INT AUTO_INCREMENT PRIMARY KEY,
    title VARCHAR(200) NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    category VARCHAR(100) DEFAULT 'general',
    tags TEXT,
    aliases TEXT,
    source_type VARCHAR(50) DEFAULT 'guidance',
    audience VARCHAR(50) DEFAULT 'student',
    priority INT DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE,
    view_count INT DEFAULT 0,
    related_questions TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Sample Data
INSERT INTO users (username, password_hash) VALUES 
('admin', '$2b$12$KIXpHOfHSy8OqicHOJAjDuwvT4.VxI4cQClfy1WKWhrHWV8JZ4y1y'); -- password: 'admin123' (use bcrypt)

-- Sample FAQs
INSERT INTO faqs (question, answer, category) VALUES
('What are the admission requirements?', 'Bachelor: 12th grade 60%+, entrance exam. Master: Bachelor 55%+, GRE. Apply via portal by deadline.', 'admissions'),
('How to register for courses?', 'Login to student portal, select semester, add courses before deadline. Max 24 credits.', 'registration'),
('What is the fee structure?', 'UG: 50k/year tuition + 5k fees. PG: 80k/year. Scholarships available.', 'fees'),
('When is the next semester start?', 'Fall: Aug 15, Spring: Jan 20. Check calendar.', 'deadlines'),
('Refund policy?', 'Full refund if withdraw within 7 days, 50% within 30 days.', 'policy'),
('Library hours?', 'Mon-Fri 8AM-10PM, Sat 10AM-6PM.', 'facilities');

-- Sample Policies
INSERT INTO policies (title, content, category) VALUES
('Academic Integrity', 'Plagiarism results in F grade and suspension. Cite sources properly.', 'policy'),
('Attendance Policy', '75% min attendance required. Medical cert for excuses.', 'policy');

-- Sample Calendar
INSERT INTO calendar (event_name, event_date, description) VALUES
('Fall Admissions Deadline', '2024-07-15', 'Last date for UG/PG applications'),
('Course Registration Opens', '2024-08-01', 'Portal opens for Fall'),
('Fall Semester Starts', '2024-08-15', 'Orientation on 14th'),
('Fee Payment Last Date', '2024-09-01', 'Late fee after this'),
('Exams Start', '2024-12-10', 'Finals week');

-- Sample Documents (handbook excerpts)
INSERT INTO documents (title, content, category) VALUES
('Student Handbook Intro', 'Welcome to University. Follow all rules for smooth experience. Contact registrar@university.edu.', 'handbook'),
('Course Registration Guide', 'Step 1: Login. Step 2: Select courses. Avoid prereqs violations.', 'handbook');

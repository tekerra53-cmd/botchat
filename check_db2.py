import sqlite3

db_path = r'C:\Users\hp\Desktop\555 university_chatbot\instance\chatbot.db'
conn = sqlite3.connect(db_path)
c = conn.cursor()

print("=== FAQs ===")
c.execute("SELECT id, question, answer, category FROM faq LIMIT 10")
for row in c.fetchall():
    print(f"Q: {row[1][:80]}")
    print(f"A: {row[2][:100]}")
    print()

print("\n=== Policies ===")
c.execute("SELECT id, title, content, category FROM policy LIMIT 5")
for row in c.fetchall():
    print(f"Title: {row[1][:80]}")
    print(f"Content: {row[2][:150]}")
    print()

print("\n=== Documents ===")
c.execute("SELECT id, title, content, category FROM document LIMIT 3")
for row in c.fetchall():
    print(f"Title: {row[1][:80]}")
    print(f"Content: {row[2][:200]}")
    print()

print("\n=== Calendar ===")
c.execute("SELECT id, event_name, event_date, description FROM calendar LIMIT 5")
for row in c.fetchall():
    print(row)

conn.close()

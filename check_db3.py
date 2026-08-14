import sqlite3

db_path = r'C:\Users\hp\Desktop\555 university_chatbot\instance\chatbot.db'
conn = sqlite3.connect(db_path)
c = conn.cursor()

print("=== All FAQs ===")
c.execute("SELECT id, question, answer FROM faq")
for row in c.fetchall():
    print(f"ID {row[0]}: {row[1][:80]}")
    print(f"  A: {row[2][:80]}")
    print()

print("\n=== All Policies ===")
c.execute("SELECT id, title, content FROM policy")
for row in c.fetchall():
    print(f"ID {row[0]}: {row[1][:80]}")
    print(f"  C: {row[2][:80]}")
    print()

conn.close()

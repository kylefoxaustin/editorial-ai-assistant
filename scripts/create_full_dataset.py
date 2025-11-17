import json
from pathlib import Path

# Here's the full 50-example dataset from your original CSV
examples = [
    {"instruction": "Edit for concision", "input": "The company's new product which was launched last week has been receiving alot of positive feedback from customers who purchased it.", "output": "The company's new product, launched last week, has received extensive positive customer feedback."},
    {"instruction": "Remove corporate jargon", "input": "We need to leverage our core competencies to synergize our value proposition and optimize stakeholder engagement moving forward.", "output": "We should focus on our strengths to improve our customer offering and better serve our stakeholders."},
    {"instruction": "Convert to active voice", "input": "The report was written by the committee and it was submitted to the board for their review and approval.", "output": "The committee wrote and submitted the report to the board for review and approval."},
    {"instruction": "Fix grammar", "input": "Its important to note that the teams performance has been effected by the recent changes in management.", "output": "It's important to note that the team's performance has been affected by recent management changes."},
    {"instruction": "Make concise", "input": "The defendant was found to be not guilty by the jury after a lengthy deliberation process.", "output": "The jury found the defendant not guilty after lengthy deliberation."},
    {"instruction": "Edit for concision", "input": "In order to achieve success in this endeavor, it is necessary for us to ensure that we have adequate resources.", "output": "To succeed, we must ensure adequate resources."},
    {"instruction": "Improve style", "input": "The meeting will be held on Tuesday, April 15th at 2:00 PM in the conference room on the third floor.", "output": "The meeting is scheduled for Tuesday, April 15, at 2:00 PM in the third-floor conference room."},
    {"instruction": "Fix grammar", "input": "John said that he would try and complete the project by the deadline if possible.", "output": "John said he would try to complete the project by the deadline."},
    {"instruction": "Improve clarity", "input": "The new policy, which was implemented last month, has already shown positive results in terms of employee satisfaction and productivity.", "output": "The new policy, implemented last month, has already improved employee satisfaction and productivity."},
    {"instruction": "Fix grammar", "input": "Each of the students need to submit their assignment by Friday.", "output": "Each student needs to submit their assignment by Friday."},
    {"instruction": "Edit for mixed improvements", "input": "The CEO announced that the company would be downsizing it's workforce by approximately 15 percent over the course of the next fiscal quarter.", "output": "The CEO announced the company would reduce its workforce by approximately 15% over the next quarter."},
    {"instruction": "Fix grammar", "input": "Between you and I, the proposal needs significant revision before it can be presented.", "output": "Between you and me, the proposal needs significant revision before presentation."},
    {"instruction": "Fix grammar", "input": "The data shows that customer satisfaction rates have increased substantially over the past six months.", "output": "The data show customer satisfaction rates have increased substantially over the past six months."},
    {"instruction": "Make concise", "input": "Due to the fact that the weather was inclement, the outdoor event had to be cancelled.", "output": "Due to inclement weather, the outdoor event was cancelled."},
    {"instruction": "Fix parallel structure", "input": "The team's strategy for winning the championship involves focusing on defense, improving their offensive capabilities, and to maintain peak physical condition.", "output": "The team's championship strategy involves focusing on defense, improving offensive capabilities, and maintaining peak physical condition."},
    {"instruction": "Fix grammar", "input": "She is one of those employees who always arrives early and stays late.", "output": "She is one of those employees who always arrive early and stay late."},
    {"instruction": "Make concise", "input": "The research indicates that there is a strong correlation between regular exercise and improved mental health outcomes.", "output": "Research indicates a strong correlation between regular exercise and improved mental health."},
    {"instruction": "Remove business cliche", "input": "Going forward, we will be implementing new procedures to ensure compliance with regulatory requirements.", "output": "We will implement new procedures to ensure regulatory compliance."},
    {"instruction": "Make concise", "input": "The manager gave the presentation to the stakeholders, and it was well-received by them.", "output": "The manager's presentation was well-received by stakeholders."},
    {"instruction": "Fix grammar", "input": "If I was in charge of the project, I would allocate resources differently.", "output": "If I were in charge of the project, I would allocate resources differently."},
    {"instruction": "Make concise", "input": "The company's mission is to provide innovative solutions that help businesses achieve their goals while maintaining sustainable practices.", "output": "The company provides innovative solutions helping businesses achieve goals through sustainable practices."},
    {"instruction": "Improve clarity", "input": "Regardless of what happens, we need to stay focused on our core objectives.", "output": "Regardless of the outcome, we must stay focused on our core objectives."},
    {"instruction": "Make concise", "input": "The novel which was written by the famous author has won several awards.", "output": "The famous author's novel has won several awards."},
    {"instruction": "Make concise", "input": "In today's fast-paced business environment, it's essential to remain agile.", "output": "Today's business environment demands agility."},
    {"instruction": "Fix grammar", "input": "The committee is comprised of seven members from various departments.", "output": "The committee comprises seven members from various departments."},
    {"instruction": "Fix grammar", "input": "Her and her team worked tirelessly to meet the deadline.", "output": "She and her team worked tirelessly to meet the deadline."},
    {"instruction": "Fix parallel structure", "input": "The software update will improve performance, enhance security features, and it will add new functionality.", "output": "The software update will improve performance, enhance security features, and add new functionality."},
    {"instruction": "Make concise", "input": "We are currently in the process of reviewing all applications.", "output": "We are reviewing all applications."},
    {"instruction": "Remove redundancy", "input": "The reason why the project failed is because of poor planning.", "output": "The project failed because of poor planning."},
    {"instruction": "Make concise", "input": "There are many factors that contribute to climate change.", "output": "Many factors contribute to climate change."},
    {"instruction": "Make concise", "input": "The meeting yesterday addressed the issue of budget overruns and how we can prevent them in the future.", "output": "Yesterday's meeting addressed budget overruns and prevention strategies."},
    {"instruction": "Make concise", "input": "Please let me know if you have any questions or concerns regarding this matter.", "output": "Please let me know if you have questions about this matter."},
    {"instruction": "Make concise", "input": "The new regulations will have an impact on how we conduct business operations.", "output": "The new regulations will impact our business operations."},
    {"instruction": "Make concise", "input": "At this point in time, we need to make a decision about the project's future.", "output": "We need to decide the project's future now."},
    {"instruction": "Make concise", "input": "The analysis of the data revealed several interesting insights into consumer behavior.", "output": "Data analysis revealed several insights into consumer behavior."},
    {"instruction": "Make concise", "input": "It is recommended that employees should submit their timesheets on a weekly basis.", "output": "Employees should submit timesheets weekly."},
    {"instruction": "Fix grammar", "input": "The manager, along with her team, are working on the proposal.", "output": "The manager, along with her team, is working on the proposal."},
    {"instruction": "Remove cliche", "input": "We need to think outside the box to find innovative solutions.", "output": "We need to find innovative solutions through creative thinking."},
    {"instruction": "Make concise", "input": "The purpose of this email is to inform you about the upcoming changes.", "output": "This email informs you about upcoming changes."},
    {"instruction": "Make concise", "input": "Despite the fact that sales were down, the company remained profitable.", "output": "Despite lower sales, the company remained profitable."},
    {"instruction": "Convert to active voice", "input": "The implementation of the new system will be done in phases.", "output": "The new system will be implemented in phases."},
    {"instruction": "Fix grammar", "input": "These type of situations require careful consideration.", "output": "These types of situations require careful consideration."},
    {"instruction": "Remove redundancy", "input": "Moving forward into the future, we need to be more proactive.", "output": "We need to be more proactive."},
    {"instruction": "Make concise", "input": "The conference will take place from March 1st to March 3rd.", "output": "The conference runs March 1-3."},
    {"instruction": "Make concise", "input": "We are pleased to announce that we have successfully completed the merger.", "output": "We have successfully completed the merger."},
    {"instruction": "Improve style", "input": "The deadline for submitting applications is March 15th, 2024.", "output": "Application deadline: March 15, 2024."},
    {"instruction": "Make concise", "input": "Research has shown that regular exercise can help to reduce stress levels.", "output": "Research shows regular exercise reduces stress."},
    {"instruction": "Make concise", "input": "The team's performance this quarter has been nothing short of exceptional.", "output": "The team performed exceptionally this quarter."},
    {"instruction": "Remove business jargon", "input": "We will circle back on this issue at a later date.", "output": "We will revisit this issue later."},
    {"instruction": "Remove redundancy", "input": "The project is behind schedule due to unforeseen circumstances beyond our control.", "output": "The project is behind schedule due to unforeseen circumstances."}
]

# 90/10 split
split_idx = int(len(examples) * 0.9)
train = examples[:split_idx]
val = examples[split_idx:]

Path('data/processed').mkdir(exist_ok=True, parents=True)

with open('data/processed/train.jsonl', 'w') as f:
    for item in train:
        f.write(json.dumps(item) + '\n')

with open('data/processed/validation.jsonl', 'w') as f:
    for item in val:
        f.write(json.dumps(item) + '\n')

print(f"Created: {len(train)} training, {len(val)} validation examples")

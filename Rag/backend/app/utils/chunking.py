from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(text, metadata=None):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    chunks = splitter.split_text(text)
    return [{"text": chunk, "metadata": dict(metadata or {})} for chunk in chunks]



#Incase if any chunks is missed in the process of chunking how to find it out 




# Page 1
# Welcome to NextGen Tech Solutions Pvt Ltd. This Human Resources Policy document defines
# the principles, rules, and structured processes that govern employment within the organization.
# Vision:
# To become a globally respected technology company known for ethical innovation, operational
# excellence,
# and employee empowerment.
# Mission:
# To design scalable, secure, and intelligent digital solutions while fostering a high-performance
# culture
# that encourages ownership, accountability, and continuous improvement.
# Core Values:
# 1. Integrity – We act with honesty and transparency in all professional dealings.
# 2. Ownership – Employees are encouraged to take responsibility beyond assigned tasks.
# 3. Innovation – Continuous experimentation and improvement are encouraged.
# 4. Collaboration – Cross-functional teamwork drives company success.
# 5. Customer Excellence – Delivering measurable value to clients is a priority.
# Organizational Structure:
# The company operates through structured departments including Engineering, DevOps, HR, Sales,
# Marketing,
# Finance, and Operations. Each department functions under defined KPIs aligned with company
# goals.

#chunk1 = Welcome to NextGen Tech Solutions Pvt Ltd. This Human Resources Policy document 
#chunk2 = This Human Resources Policy document definesthe principles, rules, and 
#chunk3 = the principles, rules, and structured processes that govern employment within the organization.


# 1 paragraph -> multiple fomrat chunks(chunking) -> his Human Resources Policy document definesthe principles, rules.. every word in this chunk is a token
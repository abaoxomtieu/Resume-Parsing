from src.graph import app
from PIL import Image
import numpy as np
from PIL import Image


graph = app

image_path = "./src/data/resume-for-fresher-template-281.jpg"
image_pil = Image.open(image_path)

# Convert the PIL image to a NumPy array
image_np = np.array(image_pil)
job_description = """
A typical **IT (Information Technology) Job Description** may look like this:

---

### **Job Title**: Information Technology (IT) Specialist / IT Support Engineer

**Location**: [City, State]

**Job Type**: Full-time

#### **Job Overview**:
The IT Specialist is responsible for maintaining the company's IT infrastructure, providing technical support, and ensuring the smooth operation of all technology-related systems. The role includes the setup, maintenance, and troubleshooting of hardware, software, and network systems.

#### **Key Responsibilities**:
- **Technical Support**: Provide helpdesk and on-site technical support to employees on software, hardware, and network issues.
- **Network Management**: Configure, monitor, and maintain local area networks (LAN), wide area networks (WAN), and VPNs.
- **System Administration**: Install, upgrade, and manage Windows/Linux servers, databases, and cloud-based environments.
- **Security**: Implement security protocols, manage firewalls, and ensure protection against cyber threats.
- **Troubleshooting**: Diagnose and resolve hardware and software issues quickly to minimize downtime.
- **Software Management**: Assist in the deployment and management of enterprise software applications such as CRM, ERP, and cloud services.
- **Documentation**: Create and maintain IT documentation and manuals, including system configurations and user guides.
- **Backup and Recovery**: Manage data backups, disaster recovery plans, and business continuity strategies.
- **Collaboration**: Work closely with cross-functional teams to implement new technologies that improve business operations.
- **Vendor Management**: Liaise with external vendors for hardware and software procurement, warranties, and services.

#### **Qualifications**:
- Bachelor's degree in Information Technology, Computer Science, or a related field.
- 1-3 years of experience in IT support, system administration, or network management.
- Strong knowledge of networking protocols (TCP/IP, DNS, DHCP).
- Experience with cloud platforms (AWS, Azure, or Google Cloud) is a plus.
- Familiarity with virtualization technologies (VMware, Hyper-V).
- Proficient in Microsoft Office 365 and/or Google Workspace administration.
- Basic knowledge of IT security practices, firewalls, and antivirus management.
- Excellent problem-solving and communication skills.
- Ability to work independently and as part of a team.

#### **Preferred Skills**:
- Certifications such as CompTIA A+, Network+, CCNA, or Microsoft Certified IT Professional (MCITP).
- Experience with scripting or automation (Python, PowerShell, or Bash).
- Knowledge of database management (SQL Server, MySQL).

---

This job description can vary depending on the company size and industry, but it outlines the general tasks and qualifications required for an IT role.

"""
input = {
    "image_origin": image_np,
    "threshold_confidence": 0.5,
    "threshold_iou": 0.5,
    "parser_output": True,
    "job_description": job_description,
    "messages": [("user", "start")]
}

output_stream = graph.stream(input, stream_mode="values")
for output in output_stream:
    if "messages" in output:
        if output["messages"][-1] is not None:
            output["messages"][-1].pretty_print()
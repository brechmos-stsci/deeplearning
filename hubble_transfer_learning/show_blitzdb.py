from blitzdb import Document
from blitzdb import FileBackend
from config import Configuration

# Setup the storage mechanism
# Load the configuration file information
c = Configuration()
data_directory = c.data_directory

# Setup the storage mechanism
backend = FileBackend("{}/prediction_database".format(data_directory))


class DataDescription(Document):
    pass


class ProcessDescription(Document):
    pass


class ProcessResult(Document):
    pass


print('Data Descriptions')
dds = backend.filter(DataDescription, {})
for dd in dds:
    print('\t', dd.name)

print('Process Descriptions')
pds = backend.filter(ProcessDescription, {})
for pd in pds:
    print('\t', pd.name)

prs = backend.filter(ProcessResult, {})
print('Process Results')
print('Ther')
for pr in prs[:20]:
    print('\t', pr.data_description.name, pr.process_description.name, pr.filename,
          pr.middle, pr.cutout_number, pr.predictions[:5])
print('There are {} entries'.format(len(prs)))
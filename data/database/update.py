import sys

#---------------------------------------------------------------------------
# Entry point to the main function of the program.
#---------------------------------------------------------------------------
if __name__ == '__main__':

	print 'updating database backup...'

	prefix = 'INSERT INTO "Image"'
	path = 'old.sql'

	sql_str = ''	
	with open(path) as sql_file:
		for line in sql_file:
			new_line = line
			if line.startswith( prefix ):
				new_line = ''
				tokens = line.split(',')
				for index, token in enumerate(tokens):
					if index > 0:
						new_line += ','
					if index == 3:
						new_line += '0,'
					new_line += token
				print new_line
			sql_str += new_line


	path = 'new.sql'
	with open(path, "w") as f:
		f.write( sql_str )

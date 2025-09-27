a.bat >> a.log
> will overwrite a.log each time you run it.
>> will append new output to the end of a.log without deleting old contents.

a.bat >> a.log 2>&1
That way both standard output and error messages are written into a.log.
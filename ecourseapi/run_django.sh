pip install -r requirement.txt

python manage.py migrate

export DJANGO_SUPERUSER_USERNAME=admin
export DJANGO_SUPERUSER_EMAIL=tn696199@gmail.com
export DJANGO_SUPERUSER_PASSWORD=123

python manage.py createsuperuser --no-input

python manage.py shell << EOF
from courses.models import Category, Course

c1, _ = Category.objects.get_or_create(name='Software Engineering')
c2, _ = Category.objects.get_or_create(name='Artificial Intelligence')
c3, _ = Category.objects.get_or_create(name='Data Sciences')

Course.objects.create(subject='Introduction to SE', description='demo', image='https://res.cloudinary.com/dxxwcby8l/image/upload/v1709565062/rohn1l6xtpxedyqgyncs.png', category=c1)
Course.objects.create(subject='Software Testing', description='demo', image='https://res.cloudinary.com/dxxwcby8l/image/upload/v1709565062/rohn1l6xtpxedyqgyncs.png', category=c1)
Course.objects.create(subject='Introduction to AI', description='demo', image='https://res.cloudinary.com/dxxwcby8l/image/upload/v1709565062/rohn1l6xtpxedyqgyncs.png', category=c2)
Course.objects.create(subject='Machine Learning', description='demo', image='https://res.cloudinary.com/dxxwcby8l/image/upload/v1709565062/rohn1l6xtpxedyqgyncs.png', category=c1)
Course.objects.create(subject='Deep Learning', description='demo', image='https://res.cloudinary.com/dxxwcby8l/image/upload/v1709565062/rohn1l6xtpxedyqgyncs.png', category=c1)
Course.objects.create(subject='Python Programming', description='demo', image='https://res.cloudinary.com/dxxwcby8l/image/upload/v1709565062/rohn1l6xtpxedyqgyncs.png', category=c3)
EOF

python manage.py runserver
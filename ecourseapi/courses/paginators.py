from rest_framework.pagination import PageNumberPagination

class CoursePaginator(PageNumberPagination):
    page_size = 5

class CommentPaginator(PageNumberPagination):
    page_size = 2

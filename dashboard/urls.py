from django.urls import path
from . import views

urlpatterns = [
    path('', views.admin_login, name='admin_login'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('logout/', views.admin_logout, name='admin_logout'),
    path('dashboard_overview/', views.dashboard_overview, name='dashboard_overview'),
    path('reset_dashboard_cache', views.reset_dashboard_cache, name='reset_dashboard_cache'),
    path('stock_prediction/', views.predict_stock, name='stock_prediction'),
    path('stock_management/', views.stock_management, name='stock_management'),
    path('reports/', views.reports_analytics, name='reports_analytics'),
    path('add_new_products/', views.add_new_products, name='add_new_products'),
    path('search_product/', views.search_product, name='search_product'),
    path('update_stock/', views.update_stock, name='update_stock'),
    path('recent_updates/', views.recent_updates, name='recent_updates'),  
    path('product_suggestions/', views.product_suggestions, name='product_suggestions'),
    path('reset_reports_cache', views.reset_reports_cache, name='reset_reports_cache'),
    path('reset_product_suggestion_cache', views.reset_product_suggestion_cache, name='reset_product_suggestion_cache'),
    path('reset_prediction_cache', views.reset_prediction_cache, name='reset_prediction_cache'),
]

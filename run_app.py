import os
import subprocess
import sys

def main():
    """
    Функция для запуска веб-приложения Streamlit
    """
    print("Запуск веб-приложения RegretNet...")
    
    # Проверяем наличие Streamlit
    try:
        import streamlit
        print(f"Streamlit версии {streamlit.__version__} найден.")
    except ImportError:
        print("Streamlit не установлен. Установка...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit>=1.15.0", "pandas>=1.3.0"])
        print("Streamlit успешно установлен.")
    
    # Запускаем приложение
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(script_dir, "app.py")
    
    if not os.path.exists(app_path):
        print(f"Ошибка: файл {app_path} не найден.")
        return
    
    print(f"Запуск приложения из {app_path}")
    print("Веб-интерфейс будет доступен по адресу: http://localhost:8501")
    
    try:
        subprocess.run(["streamlit", "run", app_path])
    except Exception as e:
        print(f"Ошибка при запуске приложения: {str(e)}")

if __name__ == "__main__":
    main() 
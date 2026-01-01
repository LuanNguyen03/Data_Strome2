#!/bin/bash
# Setup Gemini API Key for Linux/Mac

echo "================================================"
echo "  Setup Gemini AI - Treatment Recommendations"
echo "================================================"
echo ""

# Check if .env already exists
if [ -f .env ]; then
    echo "[WARNING] File .env da ton tai."
    echo ""
    read -p "Ban co muon ghi de? (y/n): " overwrite
    if [[ ! "$overwrite" =~ ^[Yy]$ ]]; then
        echo ""
        echo "Huy bo. Khong thay doi gi."
        exit 0
    fi
fi

echo ""
echo "[STEP 1] Lay API Key tu Google AI Studio"
echo "  1. Truy cap: https://makersuite.google.com/app/apikey"
echo "  2. Dang nhap bang tai khoan Google"
echo "  3. Nhan 'Create API Key' hoac 'Get API Key'"
echo "  4. Copy API key (dang: AIzaSy...)"
echo ""
echo "[STEP 2] Nhap API Key duoi day:"
read -p "GEMINI_API_KEY = " api_key

if [ -z "$api_key" ]; then
    echo ""
    echo "[ERROR] API key khong duoc de trong!"
    exit 1
fi

echo ""
echo "[STEP 3] Dang tao file .env..."
echo "GEMINI_API_KEY=$api_key" > .env

if [ $? -ne 0 ]; then
    echo "[ERROR] Khong the tao file .env"
    exit 1
fi

# Set permissions
chmod 600 .env

echo ""
echo "[SUCCESS] Da tao file .env thanh cong!"
echo ""
echo "================================================"
echo "   CAU HINH HOAN TAT"
echo "================================================"
echo ""
echo "File .env da duoc tao voi noi dung:"
echo "  GEMINI_API_KEY=$api_key"
echo ""
echo "[BUOC TIEP THEO]"
echo "  1. Restart backend (neu dang chay):"
echo "     - Nhan Ctrl+C de dung backend"
echo "     - Chay lai: uv run python backend/run.py"
echo ""
echo "  2. Kiem tra log khi backend khoi dong:"
echo "     - Nen thay: 'Gemini AI service ENABLED'"
echo ""
echo "  3. Thuc hien danh gia tu frontend"
echo "     - Section moi: 'Huong dieu tri de xuat (AI)'"
echo ""
echo "Xem huong dan chi tiet: GEMINI_SETUP.md"
echo ""

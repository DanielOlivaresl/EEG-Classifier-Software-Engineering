import QtQuick 2.15
import QtQuick.Controls 2.15

Rectangle {
    id: parentRect
    objectName: "parentRect"  // Set object name
    color: "#1a1a2e"
    anchors.fill: parent

    // Connected Button Container
    Rectangle {
        id: rectangle1
        objectName: "rectangle1"  // Set object name
        color: "#39FF14"
        radius: 10
        width: parent.width * 0.5
        height: parent.height * 0.2

        anchors.horizontalCenter: parent.horizontalCenter
        anchors.verticalCenter: parent.verticalCenter

        Text {
            id: text1
            objectName: "text1"  // Set object name
            anchors.centerIn: parent
            text: qsTr("Connected")
            font.family: "Open Sans"
            font.pixelSize: 40
            font.bold: true
            color: 'white'
        }
    }

    // Centered rectangle2, placed slightly below rectangle1
    Rectangle {
        id: rectangle2
        objectName: "rectangle2"  // Set object name
        width: 300
        height: 100
        color: "#1d2951"
        border.color: '#1E90FF'
        border.width: 3
        radius: 10

        anchors.horizontalCenter: parent.horizontalCenter
        anchors.top: rectangle1.bottom
        anchors.topMargin: 20

        Text {
            id: text2
            objectName: "text2"  // Set object name
            anchors.centerIn: parent
            text: qsTr("Text")
            font.family: "Open Sans"
            font.pixelSize: 50
            font.bold: true
            color: "white"
        }
    }
}

<?php

// Define a whitelist of allowed files
$whitelist = ['home.html', 'contact.html', 'creditcard','.htpasswd','.htaccess'];

// Get the requested page from user input
$page = $_REQUEST['page'];

// Check if the requested page is in the whitelist
if (in_array($page, $whitelist)) {
    echo "File included: " . htmlspecialchars($page) . "<br>";
    echo "<br><br>";
    $local_file = $page;
    echo "Local file to be used: " . htmlspecialchars($local_file);
    echo "<br><br>";

    include $local_file;
} else {
    echo "Invalid file request.";
}

?>